# main.py
# Main FastAPI application file

import os
import shutil
import logging
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn as nn # For Autoencoder
from transformers import (
    AutoTokenizer,
    AutoModel, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
from sentence_transformers import SentenceTransformer 
from datasets import load_dataset, Dataset, DatasetDict, Sequence
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import numpy as np
import pandas as pd

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PyTorch Autoencoder Definition ---
class TextEmbeddingAutoencoder(nn.Module):
    def __init__(self, embedding_dim: int, encoding_dim: int):
        super(TextEmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(embedding_dim / 2), encoding_dim),
            nn.ReLU() 
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, int(embedding_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(embedding_dim / 2), embedding_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Pydantic Models for API ---
class InferenceRequest(BaseModel):
    task: str = Field(..., description="NLP task")
    model_name: str = Field(..., description="Hugging Face model identifier")
    input_text: Union[str, List[str]] = Field(..., description="Text to process")
    context: Optional[str] = Field(None, description="Context for tasks like Question Answering")
    quantization: Optional[str] = Field("none", description="Quantization: none, dynamic_int8_cpu")
    generation_args: Optional[Dict[str, Any]] = Field(None, description="Arguments for text generation or summarization")

class InferenceResponse(BaseModel):
    predictions: Any
    model_used: str
    quantization_applied: str

class LoRAConfigModel(BaseModel):
    r: int = Field(8)
    alpha: int = Field(16)
    dropout: float = Field(0.05)
    target_modules: Optional[Union[str, List[str]]] = Field("q_proj,v_proj")

class FineTuneParamsModel(BaseModel):
    dataset_path: str = Field(...)
    text_column: Optional[str] = Field("text", description="Name of the main text/document/tokens column")
    label_column: Optional[str] = Field("label", description="Name of the target/summary/label/tags column")
    context_column: Optional[str] = Field("context", description="Name of the context column for QA fine-tuning")
    epochs: int = Field(3)
    batch_size: int = Field(8)
    learning_rate: float = Field(2e-5)
    use_lora: bool = Field(True)
    lora_config: Optional[LoRAConfigModel] = None
    output_dir_base: str = Field("./finetuned_models")
    max_seq_length: int = Field(384) 
    max_target_length: Optional[int] = Field(128, description="Max length for target sequences in seq2seq tasks")
    num_labels: Optional[int] = None

class FineTuneRequest(BaseModel):
    task: str = Field(...)
    model_name: str = Field(...)
    fine_tune_params: FineTuneParamsModel

class FineTuneResponse(BaseModel):
    message: str
    model_name: str
    adapter_output_path: Optional[str] = None
    logs: Optional[str] = None

class AnomalyDetectionRequest(BaseModel):
    text: str = Field(..., description="Input text to check for anomaly")
    embedding_model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Sentence Transformer model for embeddings")
    autoencoder_model_path: str = Field(..., description="Path to the saved PyTorch autoencoder state_dict")
    autoencoder_embedding_dim: int = Field(384, description="Embedding dimension the AE was trained on (e.g., 384 for all-MiniLM-L6-v2)")
    autoencoder_encoding_dim: int = Field(64, description="Bottleneck encoding dimension of the AE")
    threshold: float = Field(0.1, description="Reconstruction error threshold for anomaly classification")

class AnomalyDetectionResponse(BaseModel):
    text: str
    is_anomaly: bool
    reconstruction_error: float 
    threshold_used: float
    embedding_model_used: str
    autoencoder_model_used: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hugging Face Model Runner API",
    description="API to run inference, fine-tune models, and detect anomalies.",
    version="0.2.1" # Version bump
)

# --- CORS Middleware Configuration ---
origins = [
    "http://localhost:3000",
    "https://yourdomain.com", 
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Global Variables / Model Cache ---
loaded_models_cache = {} 
loaded_embedding_models_cache = {} 
loaded_autoencoders_cache = {} 

# --- Helper Functions ---

def sanitize_value_for_json(value: Any) -> Any:
    """
    Recursively sanitizes values that might be NumPy types to native Python types for JSON serialization.
    """
    if isinstance(value, list):
        return [sanitize_value_for_json(item) for item in value]
    elif isinstance(value, dict):
        return {k: sanitize_value_for_json(v) for k, v in value.items()}
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.int32, np.int64, np.int_)):
        return int(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray): # Handle arrays, convert to list
        return sanitize_value_for_json(value.tolist())
    elif isinstance(value, np.generic): # Catch-all for other numpy scalar types
        return value.item()
    return value


def get_embedding_model(model_name: str):
    if model_name in loaded_embedding_models_cache:
        logger.info(f"Using cached embedding model: {model_name}")
        return loaded_embedding_models_cache[model_name]
    
    logger.info(f"Loading sentence embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name, device='cpu')
        loaded_embedding_models_cache[model_name] = model
        logger.info(f"Embedding model {model_name} loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading sentence embedding model {model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not load sentence embedding model: {model_name}")

def load_autoencoder_model(path: str, embedding_dim: int, encoding_dim: int):
    cache_key = (path, embedding_dim, encoding_dim)
    if cache_key in loaded_autoencoders_cache:
        logger.info(f"Using cached autoencoder model: {path}")
        return loaded_autoencoders_cache[cache_key]

    logger.info(f"Loading autoencoder model from: {path}...")
    if not os.path.exists(path):
        logger.error(f"Autoencoder model file not found at: {path}")
        raise HTTPException(status_code=404, detail=f"Autoencoder model not found at path: {path}")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        autoencoder = TextEmbeddingAutoencoder(embedding_dim, encoding_dim).to(device)
        autoencoder.load_state_dict(torch.load(path, map_location=device))
        autoencoder.eval() 
        loaded_autoencoders_cache[cache_key] = autoencoder
        logger.info(f"Autoencoder model {path} loaded successfully to {device}.")
        return autoencoder
    except Exception as e:
        logger.error(f"Error loading autoencoder model {path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not load autoencoder model: {path}")

def get_model_and_tokenizer(model_name: str, task: str, num_labels: Optional[int] = None, quantization: str = "none"):
    cache_key_num_labels_suffix = None
    if task == "text-classification" or task == "token-classification":
        cache_key_num_labels_suffix = num_labels

    cache_key = (model_name, quantization, task, cache_key_num_labels_suffix)
    if cache_key in loaded_models_cache:
        logger.info(f"Using cached model: {model_name} ({quantization}, task: {task}, num_labels for cache key: {cache_key[3]})")
        return loaded_models_cache[cache_key]

    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Loading model {model_name} for task {task}...")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.info(f"Tokenizer for {model_name} has no pad_token. Setting pad_token to eos_token ({tokenizer.eos_token}).")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning(f"Tokenizer for {model_name} has no pad_token and no eos_token. Adding a new pad_token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model_to_load = None
    model_args = {}
    if task == "text-classification" or task == "token-classification":
        if num_labels is not None:
            logger.info(f"Loading {model_name} for {task} with explicit num_labels: {num_labels}")
            model_args["num_labels"] = num_labels
            model_args["ignore_mismatched_sizes"] = True 
        else:
            logger.info(f"Loading {model_name} for {task} inference (num_labels will be inferred from model config).")
        
        if task == "text-classification":
            model_to_load = AutoModelForSequenceClassification.from_pretrained(model_name, **model_args)
        elif task == "token-classification":
            model_to_load = AutoModelForTokenClassification.from_pretrained(model_name, **model_args)

    elif task == "text-generation":
        model_to_load = AutoModelForCausalLM.from_pretrained(model_name)
    elif task == "question-answering":
        logger.info(f"Loading {model_name} for question-answering.")
        model_to_load = AutoModelForQuestionAnswering.from_pretrained(model_name)
    elif task == "summarization" or task == "translation":
        logger.info(f"Loading {model_name} for {task} (Seq2Seq).")
        model_to_load = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task for model loading: {task}")
    
    if model_to_load is None: 
        raise HTTPException(status_code=500, detail=f"Model could not be loaded for task {task}.")

    if tokenizer.pad_token_id is not None and \
       hasattr(model_to_load, 'config') and \
       hasattr(model_to_load.config, 'vocab_size') and \
       tokenizer.pad_token_id >= model_to_load.config.vocab_size and \
       hasattr(model_to_load, 'resize_token_embeddings'):
        logger.info(f"Resizing token embeddings for {model_name} due to new pad_token.")
        model_to_load.resize_token_embeddings(len(tokenizer))

    model_to_load.eval()

    if quantization == "dynamic_int8_cpu":
        if not hasattr(torch.quantization, "quantize_dynamic"):
             raise HTTPException(status_code=501, detail="Dynamic quantization unavailable.")
        logger.info(f"Applying dynamic INT8 quantization to {model_name} for CPU...")
        model_to_load = model_to_load.to("cpu")
        model_to_load = torch.quantization.quantize_dynamic(model_to_load, {torch.nn.Linear}, dtype=torch.qint8)
        logger.info("Dynamic quantization applied.")
    elif quantization != "none":
        logger.warning(f"Unsupported quantization: {quantization}. Using none.")
    
    loaded_models_cache[cache_key] = (model_to_load, tokenizer)
    return model_to_load, tokenizer

def get_lora_target_modules(model_name_or_path: str, user_specified_modules: Optional[Union[str, List[str]]]) -> List[str]:
    if isinstance(user_specified_modules, str):
        modules = [m.strip() for m in user_specified_modules.split(",") if m.strip()]
        if modules: return modules
    elif isinstance(user_specified_modules, list) and user_specified_modules:
        return user_specified_modules
    
    logger.info("Inferring LoRA target modules.")
    name = model_name_or_path.lower()
    if "t5" in name or "bart" in name or "pegasus" in name: 
        return ["q", "v", "k", "o", "wi", "wo"] 
    if "distilbert" in name: return ["q_lin", "k_lin", "v_lin", "out_lin"]
    if "bert" in name or "roberta" in name or "electra" in name: 
        return ["query", "key", "value", "dense"] 
    if any(n in name for n in ["gpt2", "llama", "mistral", "opt"]):
        if "llama" in name or "mistral" in name: return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if "gpt2" in name: return ["c_attn", "c_proj"] 
        return ["q_proj", "v_proj"] 
    logger.warning(f"Could not infer LoRA targets for {name}.")
    return []

# --- API Endpoints ---
@app.post("/api/v1/infer", response_model=InferenceResponse)
async def run_inference_endpoint(request: InferenceRequest):
    logger.info(f"Received inference request: Task={request.task}, Model={request.model_name}")
    try:
        model, tokenizer = get_model_and_tokenizer(
            model_name=request.model_name,
            task=request.task,
            quantization=request.quantization
        )

        raw_predictions = None # Store raw pipeline output
        if request.task == "text-classification":
            classifier = pipeline(
                "sentiment-analysis" if "sentiment" in request.model_name.lower() or "sst-2" in request.model_name.lower() else "text-classification",
                model=model, tokenizer=tokenizer, device=-1 
            )
            results = classifier(request.input_text if isinstance(request.input_text, list) else [request.input_text])
            raw_predictions = results

        elif request.task == "text-generation":
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None: tokenizer.pad_token_id = tokenizer.eos_token_id
                else: raise ValueError("Tokenizer has no pad_token_id and no eos_token_id for generation padding.")
            
            inputs = tokenizer(request.input_text, return_tensors="pt", padding=True, truncation=True)
            gen_args = request.generation_args or {}
            default_gen_args = {"max_length": 50, "num_return_sequences": 1, "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
            final_gen_args = {**default_gen_args, **gen_args}
            
            with torch.no_grad(): outputs = model.generate(**inputs, **final_gen_args)
            
            decoded_outputs = []
            if isinstance(request.input_text, list) and len(outputs) >= len(inputs["input_ids"]):
                num_return_sequences = final_gen_args.get("num_return_sequences", 1)
                num_inputs = len(inputs["input_ids"])
                for i in range(num_inputs):
                    batch_predictions_texts = []
                    for j in range(num_return_sequences):
                        output_idx = i * num_return_sequences + j
                        if output_idx < len(outputs):
                           batch_predictions_texts.append(tokenizer.decode(outputs[output_idx], skip_special_tokens=True))
                    decoded_outputs.append(batch_predictions_texts[0] if num_return_sequences == 1 else batch_predictions_texts)
            elif not isinstance(request.input_text, list):
                decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else: 
                logger.warning("Decoding multiple sequences for batched text generation, structure might vary.")
                decoded_outputs = [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in outputs]
            raw_predictions = decoded_outputs
        
        elif request.task == "question-answering":
            if not request.context:
                raise HTTPException(status_code=400, detail="Context is required for question-answering task.")
            if not isinstance(request.input_text, str):
                raise HTTPException(status_code=400, detail="Input text for question-answering should be a single question string.")

            qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)
            results = qa_pipeline(question=request.input_text, context=request.context)
            raw_predictions = results

        elif request.task == "summarization":
            if not request.input_text:
                raise HTTPException(status_code=400, detail="Input text is required for summarization.")
            
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
            gen_args = request.generation_args or {}
            default_summary_args = {"min_length": 30, "max_length": 150, "num_beams": 4, "early_stopping": True}
            final_summary_args = {**default_summary_args, **gen_args}

            if isinstance(request.input_text, str):
                results = summarizer(request.input_text, **final_summary_args)
                # Pipeline returns a list with one dict for single string input
                raw_predictions = results[0] if results else None 
            elif isinstance(request.input_text, list):
                raw_predictions = summarizer(request.input_text, **final_summary_args)
            else:
                raise HTTPException(status_code=400, detail="Invalid input_text format for summarization.")

        elif request.task == "translation": 
            if not request.input_text:
                raise HTTPException(status_code=400, detail="Input text is required for translation.")
            translator = pipeline("translation", model=model, tokenizer=tokenizer, device=-1)
            gen_args = request.generation_args or {}
            if isinstance(request.input_text, str):
                results = translator(request.input_text, **gen_args)
                raw_predictions = results[0] if results else None
            elif isinstance(request.input_text, list):
                raw_predictions = translator(request.input_text, **gen_args)
            else:
                raise HTTPException(status_code=400, detail="Invalid input_text format for translation.")
        
        elif request.task == "token-classification":
            if not request.input_text:
                raise HTTPException(status_code=400, detail="Input text is required for token classification.")
            
            token_classifier = pipeline(
                "ner", model=model, tokenizer=tokenizer, device=-1, grouped_entities=True 
            )
            if isinstance(request.input_text, str):
                raw_predictions = token_classifier(request.input_text)
            elif isinstance(request.input_text, list):
                raw_predictions = token_classifier(request.input_text)
            else:
                raise HTTPException(status_code=400, detail="Invalid input_text format for token classification.")
        else:
            raise HTTPException(status_code=400, detail=f"Inference for task '{request.task}' not yet implemented.")

        # Sanitize predictions before returning
        sanitized_predictions = sanitize_value_for_json(raw_predictions)

        return InferenceResponse(
            predictions=sanitized_predictions, 
            model_used=request.model_name, 
            quantization_applied=request.quantization
        )
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- New Anomaly Detection Endpoint ---
@app.post("/api/v1/detect_anomaly", response_model=AnomalyDetectionResponse)
async def detect_text_anomaly(request: AnomalyDetectionRequest):
    logger.info(f"Received anomaly detection request for text: '{request.text[:50]}...'")
    logger.info(f"Using embedding model: {request.embedding_model_name}")
    logger.info(f"Using autoencoder model: {request.autoencoder_model_path}")

    try:
        embedding_model = get_embedding_model(request.embedding_model_name)
        autoencoder = load_autoencoder_model(
            request.autoencoder_model_path,
            request.autoencoder_embedding_dim,
            request.autoencoder_encoding_dim
        )
        device = next(autoencoder.parameters()).device 

        logger.info("Generating text embedding...")
        input_embedding_np = embedding_model.encode(request.text)
        input_embedding = torch.tensor(input_embedding_np, dtype=torch.float32).unsqueeze(0).to(device) 

        logger.info("Getting reconstruction from autoencoder...")
        with torch.no_grad():
            reconstructed_embedding = autoencoder(input_embedding)

        loss_fn = nn.MSELoss()
        reconstruction_error_tensor = loss_fn(reconstructed_embedding, input_embedding)
        reconstruction_error_float = float(reconstruction_error_tensor.item()) 
        logger.info(f"Reconstruction error: {reconstruction_error_float}")

        is_anomaly = reconstruction_error_float > request.threshold
        logger.info(f"Is anomaly: {is_anomaly} (Threshold: {request.threshold})")

        return AnomalyDetectionResponse(
            text=request.text,
            is_anomaly=is_anomaly,
            reconstruction_error=reconstruction_error_float, 
            threshold_used=float(request.threshold), # Ensure threshold is also Python float
            embedding_model_used=request.embedding_model_name,
            autoencoder_model_used=request.autoencoder_model_path
        )

    except HTTPException: 
        raise
    except Exception as e:
        logger.error(f"Error during anomaly detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@app.post("/api/v1/finetune", response_model=FineTuneResponse)
async def run_fine_tuning_endpoint(request: FineTuneRequest, background_tasks: BackgroundTasks):
    params = request.fine_tune_params
    logger.info(f"Fine-tune: Task={request.task}, Model={request.model_name}, Data={params.dataset_path}")
    run_name = f"{request.model_name.replace('/', '_')}_{request.task}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(params.output_dir_base, run_name)
    os.makedirs(output_dir, exist_ok=True)
    adapter_output_path = os.path.join(output_dir, "lora_adapters")

    try:
        tokenizer = AutoTokenizer.from_pretrained(request.model_name)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
            else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        if params.dataset_path.endswith(".csv"):
            df = pd.read_csv(params.dataset_path)
            required_csv_cols = [params.text_column]
            if request.task == "text-classification": required_csv_cols.append(params.label_column)
            elif request.task == "question-answering": required_csv_cols.extend([params.context_column, params.label_column])
            elif request.task == "summarization" or request.task == "translation": required_csv_cols.append(params.label_column)
            elif request.task == "token-classification": required_csv_cols.append(params.label_column)
            
            for col in required_csv_cols:
                if col not in df.columns:
                    raise HTTPException(status_code=400, detail=f"Column '{col}' not found in CSV for task '{request.task}'.")
            train_df = df.sample(frac=0.8, random_state=42); eval_df = df.drop(train_df.index)
            raw_datasets = DatasetDict({"train": Dataset.from_pandas(train_df), "validation": Dataset.from_pandas(eval_df)})
        else:
            raw_datasets = load_dataset(params.dataset_path)

        train_split, eval_split = "train", "validation" if "validation" in raw_datasets else "test" if "test" in raw_datasets else None
        if train_split not in raw_datasets: raise HTTPException(status_code=400, detail=f"'{train_split}' not in dataset.")
        
        max_seq_len, max_target_len = params.max_seq_length, params.max_target_length

        def preprocess_seq2seq(examples):
            model_inputs = tokenizer(examples[params.text_column], max_length=max_seq_len, truncation=True, padding="max_length")
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples[params.label_column], max_length=max_target_len, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        def tokenize_fn_qa(examples):
            questions = [q.strip() for q in examples[params.text_column]] 
            contexts = examples[params.context_column]
            inputs = tokenizer(questions, contexts, max_length=max_seq_len, truncation="only_second", stride=128, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
            offset_mapping, sample_mapping, answers = inputs.pop("offset_mapping"), inputs.pop("overflow_to_sample_mapping"), examples[params.label_column] 
            start_positions, end_positions = [], []
            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_mapping[i]; answer = answers[sample_idx]
                if not answer['answer_start']: start_positions.append(0); end_positions.append(0); continue
                start_char, end_char = answer['answer_start'][0], answer['answer_start'][0] + len(answer['text'][0])
                sequence_ids = inputs.sequence_ids(i)
                context_start_idx, context_end_idx = sequence_ids.index(1), len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
                if not (offset[context_start_idx][0] <= start_char and offset[context_end_idx][1] >= end_char):
                    start_positions.append(0); end_positions.append(0)
                else:
                    token_start_index = context_start_idx
                    while token_start_index <= context_end_idx and offset[token_start_index][0] <= start_char: token_start_index += 1
                    start_positions.append(token_start_index - 1)
                    token_end_index = context_end_idx
                    while token_end_index >= context_start_idx and offset[token_end_index][1] >= end_char: token_end_index -= 1
                    end_positions.append(token_end_index + 1)
            inputs["start_positions"], inputs["end_positions"] = start_positions, end_positions
            return inputs

        def tokenize_fn_generic(ex): 
            if params.text_column not in ex: raise ValueError(f"Text column '{params.text_column}' missing.")
            tk_batch = tokenizer(ex[params.text_column], truncation=True, padding="max_length", max_length=max_seq_len)
            if params.label_column not in ex: raise ValueError(f"Label column '{params.label_column}' missing.")
            tk_batch["labels"] = ex[params.label_column]
            return tk_batch

        def tokenize_and_align_labels_for_ner(examples):
            tokenized_inputs = tokenizer(
                examples[params.text_column], truncation=True, is_split_into_words=True, 
                padding="max_length", max_length=max_seq_len
            )
            labels = []
            for i, label_list in enumerate(examples[params.label_column]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx, label_ids = None, []
                for word_idx in word_ids:
                    if word_idx is None: label_ids.append(-100)
                    elif word_idx != previous_word_idx: label_ids.append(label_list[word_idx] if isinstance(label_list[word_idx], int) else -100)
                    else: label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        chosen_tokenize_fn = tokenize_fn_generic
        if request.task == "question-answering": chosen_tokenize_fn = tokenize_fn_qa
        elif request.task == "summarization" or request.task == "translation": chosen_tokenize_fn = preprocess_seq2seq
        elif request.task == "token-classification":
            logger.warning("Using simplified placeholder for NER tokenization. For effective NER fine-tuning, implement proper label alignment.")
            chosen_tokenize_fn = tokenize_and_align_labels_for_ner 

        cols_to_remove_for_map = list(raw_datasets[train_split].column_names)
        tokenized_ds = raw_datasets.map(chosen_tokenize_fn, batched=True, remove_columns=cols_to_remove_for_map)
        
        num_labels_for_task = params.num_labels
        if (request.task == "text-classification" or request.task == "token-classification") and not num_labels_for_task:
            if "labels" not in tokenized_ds[train_split].features: raise HTTPException(status_code=400, detail="'labels' not in tokenized data.")
            labels_feature = tokenized_ds[train_split].features.get("labels")
            if request.task == "token-classification" and isinstance(labels_feature, Sequence) and hasattr(labels_feature.feature, "num_classes"):
                num_labels_for_task = labels_feature.feature.num_classes
            elif hasattr(labels_feature, "num_classes"): 
                num_labels_for_task = labels_feature.num_classes
            else: 
                if request.task == "text-classification":
                    unique_labels_list = tokenized_ds[train_split].unique("labels")
                    if unique_labels_list: num_labels_for_task = len(unique_labels_list)
            if not num_labels_for_task: raise HTTPException(status_code=400, detail="num_labels needed and not inferable for " + request.task)
            logger.info(f"Inferred num_labels for {request.task}: {num_labels_for_task}")

        base_model_args = {}
        ModelClass = None
        if request.task == "text-classification": ModelClass, base_model_args["num_labels"], base_model_args["ignore_mismatched_sizes"] = AutoModelForSequenceClassification, num_labels_for_task, True
        elif request.task == "token-classification": ModelClass, base_model_args["num_labels"], base_model_args["ignore_mismatched_sizes"] = AutoModelForTokenClassification, num_labels_for_task, True
        elif request.task == "text-generation": ModelClass = AutoModelForCausalLM
        elif request.task == "question-answering": ModelClass = AutoModelForQuestionAnswering
        elif request.task == "summarization" or request.task == "translation": ModelClass = AutoModelForSeq2SeqLM
        
        if not ModelClass: raise HTTPException(status_code=400, detail=f"Fine-tuning for task {request.task} unsupported.")
        base_model = ModelClass.from_pretrained(request.model_name, **base_model_args)
        
        if tokenizer.pad_token_id is not None and hasattr(base_model, 'resize_token_embeddings') and tokenizer.pad_token_id >= base_model.config.vocab_size:
             base_model.resize_token_embeddings(len(tokenizer))

        peft_model = base_model
        if params.use_lora and params.lora_config:
            lora_c, task_type_peft = params.lora_config, None
            if request.task == "text-classification": task_type_peft = TaskType.SEQ_CLS
            elif request.task == "token-classification": task_type_peft = TaskType.TOKEN_CLS
            elif request.task == "text-generation": task_type_peft = TaskType.CAUSAL_LM
            elif request.task == "question-answering": task_type_peft = TaskType.QUESTION_ANS
            elif request.task == "summarization" or request.task == "translation": task_type_peft = TaskType.SEQ_2_SEQ_LM
            
            if not task_type_peft: raise HTTPException(status_code=400, detail=f"LoRA task type undetermined for {request.task}.")
            lora_targets = get_lora_target_modules(request.model_name, lora_c.target_modules)
            if not lora_targets: logger.warning("No LoRA targets; LoRA may not be effective.")
            
            peft_config = LoraConfig(r=lora_c.r, lora_alpha=lora_c.alpha, lora_dropout=lora_c.dropout, target_modules=lora_targets, bias="none", task_type=task_type_peft)
            peft_model = get_peft_model(base_model, peft_config)
            peft_model.print_trainable_parameters()

        training_args_dict = {
            "output_dir": output_dir, "num_train_epochs": params.epochs,
            "per_device_train_batch_size": params.batch_size, "per_device_eval_batch_size": params.batch_size,
            "learning_rate": params.learning_rate, "weight_decay": 0.01,
            "logging_dir": f"{output_dir}/logs", "logging_steps": max(1, int(len(tokenized_ds[train_split]) / (params.batch_size * 10))),
            "evaluation_strategy": "epoch" if eval_split and eval_split in tokenized_ds else "no",
            "save_strategy": "epoch" if eval_split and eval_split in tokenized_ds else "steps",
            "save_total_limit": 2, "load_best_model_at_end": bool(eval_split and eval_split in tokenized_ds),
            "no_cuda": True, "report_to": "none",
            "predict_with_generate": True if request.task in ["summarization", "translation"] else False 
        }
        training_args = TrainingArguments(**training_args_dict)
        
        data_collator_for_trainer = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized_ds[train_split], eval_dataset=tokenized_ds.get(eval_split), tokenizer=tokenizer, data_collator=data_collator_for_trainer)
        
        logger.info("Starting fine-tuning...")
        trainer.train()
        logger.info("Fine-tuning finished.")
        
        final_save_path = adapter_output_path if params.use_lora else os.path.join(output_dir, "full_model_final")
        if params.use_lora: peft_model.save_pretrained(final_save_path)
        else: trainer.save_model(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        
        for item in os.listdir(output_dir): 
            if os.path.isdir(p := os.path.join(output_dir, item)) and item.startswith("checkpoint-"):
                logger.info(f"Removing intermediate checkpoint: {p}")
                shutil.rmtree(p)
        
        return FineTuneResponse(message="Fine-tuning completed.", model_name=request.model_name, adapter_output_path=final_save_path, logs=f"Logs in {output_dir}")
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Fine-tuning error: {e}", exc_info=True)
        if os.path.exists(output_dir): shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {str(e)}")

@app.get("/")
async def root(): return {"message": "Welcome to HF Model Runner API!"}
