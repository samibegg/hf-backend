# main.py
# Main FastAPI application file

import os
import shutil
import logging
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering, # Added for QA
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import numpy as np
import pandas as pd

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---
class InferenceRequest(BaseModel):
    task: str = Field(..., description="NLP task")
    model_name: str = Field(..., description="Hugging Face model identifier")
    input_text: Union[str, List[str]] = Field(..., description="Text to process (e.g., question for QA, sentence for classification)")
    context: Optional[str] = Field(None, description="Context for tasks like Question Answering") # Added for QA
    quantization: Optional[str] = Field("none", description="Quantization: none, dynamic_int8_cpu")
    generation_args: Optional[Dict[str, Any]] = None

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
    text_column: Optional[str] = Field("text") # For QA, this could be 'question'
    label_column: Optional[str] = Field("label") # For QA, this could be 'answers' or split into 'answer_start', 'text'
    context_column: Optional[str] = Field("context", description="Name of the context column for QA fine-tuning") # Added for QA fine-tuning
    epochs: int = Field(3)
    batch_size: int = Field(8)
    learning_rate: float = Field(2e-5)
    use_lora: bool = Field(True)
    lora_config: Optional[LoRAConfigModel] = None
    output_dir_base: str = Field("./finetuned_models")
    max_seq_length: int = Field(384) # QA often uses longer sequences
    num_labels: Optional[int] = None # Not typically used for QA model loading itself

class FineTuneRequest(BaseModel):
    task: str = Field(...)
    model_name: str = Field(...)
    fine_tune_params: FineTuneParamsModel

class FineTuneResponse(BaseModel):
    message: str
    model_name: str
    adapter_output_path: Optional[str] = None
    logs: Optional[str] = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hugging Face Model Runner API",
    description="API to run inference and fine-tune Hugging Face models.",
    version="0.1.0"
)

# --- CORS Middleware Configuration ---
origins = [
    "http://localhost:3000",
    "https://yourdomain.com", # Replace with your frontend's production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables / Model Cache ---
loaded_models_cache = {}

# --- Helper Functions ---
def get_model_and_tokenizer(model_name: str, task: str, num_labels: Optional[int] = None, quantization: str = "none"):
    cache_key = (model_name, quantization, task, num_labels if task == "text-classification" else None)
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

    if task == "text-classification":
        if num_labels is not None:
            logger.info(f"Loading {model_name} for text-classification with explicit num_labels: {num_labels}")
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        else:
            logger.info(f"Loading {model_name} for text-classification inference (num_labels will be inferred from model config).")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task == "text-generation":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= model.config.vocab_size:
             logger.info(f"Resizing token embeddings for {model_name} due to new pad_token.")
             model.resize_token_embeddings(len(tokenizer))
    elif task == "question-answering": # Added QA model loading
        logger.info(f"Loading {model_name} for question-answering.")
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task for model loading: {task}")
    
    model.eval()

    if quantization == "dynamic_int8_cpu":
        if not hasattr(torch.quantization, "quantize_dynamic"):
             raise HTTPException(status_code=501, detail="Dynamic quantization unavailable.")
        logger.info(f"Applying dynamic INT8 quantization to {model_name} for CPU...")
        model = model.to("cpu")
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        logger.info("Dynamic quantization applied.")
    elif quantization != "none":
        logger.warning(f"Unsupported quantization: {quantization}. Using none.")
    
    loaded_models_cache[cache_key] = (model, tokenizer)
    return model, tokenizer

def get_lora_target_modules(model_name_or_path: str, user_specified_modules: Optional[Union[str, List[str]]]) -> List[str]:
    if isinstance(user_specified_modules, str):
        modules = [m.strip() for m in user_specified_modules.split(",") if m.strip()]
        if modules: return modules
    elif isinstance(user_specified_modules, list) and user_specified_modules:
        return user_specified_modules
    
    logger.info("Inferring LoRA target modules.")
    name = model_name_or_path.lower()
    if "distilbert" in name: return ["q_lin", "k_lin", "v_lin", "out_lin"] # Specific to DistilBertSelfAttention
    if "bert" in name and "roberta" not in name : return ["query", "key", "value", "dense"] # BertSelfAttention, BertIntermediate, BertOutput
    if "roberta" in name: return ["query", "key", "value", "dense"] # RobertaSelfAttention, RobertaIntermediate, RobertaOutput
    if any(n in name for n in ["gpt2", "llama", "mistral", "opt"]):
        logger.warning(f"Generic LoRA targets for {name}. Verify for optimal targets (e.g., q_proj, v_proj for Llama; c_attn for GPT-2).")
        # Common patterns, but highly model-specific
        if "llama" in name or "mistral" in name: return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if "gpt2" in name: return ["c_attn", "c_proj"] 
        return ["q_proj", "v_proj"] # A very general guess
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

        if request.task == "text-classification":
            classifier = pipeline(
                "sentiment-analysis" if "sentiment" in request.model_name.lower() or "sst-2" in request.model_name.lower() else "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1 
            )
            results = classifier(request.input_text if isinstance(request.input_text, list) else [request.input_text])
            predictions = results

        elif request.task == "text-generation":
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None: tokenizer.pad_token_id = tokenizer.eos_token_id
                else: raise ValueError("Tokenizer has no pad_token_id and no eos_token_id for generation padding.")
            
            inputs = tokenizer(request.input_text, return_tensors="pt", padding=True, truncation=True)
            gen_args = request.generation_args or {}
            final_gen_args = {**{"max_length": 50, "num_return_sequences": 1}, **gen_args}
            with torch.no_grad(): outputs = model.generate(**inputs, **final_gen_args)
            
            if isinstance(request.input_text, list) and len(outputs) == len(inputs["input_ids"]) * final_gen_args.get("num_return_sequences", 1) :
                predictions = [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in outputs]
            elif not isinstance(request.input_text, list):
                predictions = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else: 
                logger.warning("Decoding multiple sequences for batched text generation, structure might vary.")
                predictions = [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in outputs]
        
        elif request.task == "question-answering": # Added QA inference
            if not request.context:
                raise HTTPException(status_code=400, detail="Context is required for question-answering task.")
            if not isinstance(request.input_text, str): # QA pipeline typically expects single question
                raise HTTPException(status_code=400, detail="Input text for question-answering should be a single question string.")

            qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=-1
            )
            # The pipeline expects question and context
            results = qa_pipeline(question=request.input_text, context=request.context)
            predictions = results # result is usually a dict with 'score', 'start', 'end', 'answer'

        else:
            raise HTTPException(status_code=400, detail=f"Inference for task '{request.task}' not yet implemented.")

        return InferenceResponse(
            predictions=predictions,
            model_used=request.model_name,
            quantization_applied=request.quantization
        )

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Fine-tuning: Set pad_token to eos_token ({tokenizer.eos_token}) for {request.model_name}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.warning(f"Fine-tuning: Added new pad_token '[PAD]' for {request.model_name}")

        if params.dataset_path.endswith(".csv"):
            df = pd.read_csv(params.dataset_path)
            if params.text_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Text column '{params.text_column}' not found in CSV.")
            if request.task == "text-classification" and params.label_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Label column '{params.label_column}' not found in CSV for text classification.")
            # For QA, ensure context and answer-related columns exist if it's a CSV
            if request.task == "question-answering":
                if params.context_column not in df.columns:
                     raise HTTPException(status_code=400, detail=f"Context column '{params.context_column}' not found in CSV for QA.")
                # QA fine-tuning usually expects answer start and text, or a specific structure.
                # This example assumes a simplified 'label_column' might hold answer text for some formats,
                # but SQuAD-like formats are more complex. This part might need significant enhancement for robust QA fine-tuning.
                # For now, we'll assume the tokenize_fn handles it.

            train_df = df.sample(frac=0.8, random_state=42)
            eval_df = df.drop(train_df.index)
            raw_datasets = DatasetDict({"train": Dataset.from_pandas(train_df), "validation": Dataset.from_pandas(eval_df)})
        else:
            raw_datasets = load_dataset(params.dataset_path) # SQuAD is often loaded this way

        # Tokenization function needs to be adapted for QA
        def tokenize_fn_qa(examples):
            # SQuAD-like preprocessing. `question` and `context` fields are expected.
            # `text_column` from params should map to 'question', `context_column` to 'context'.
            questions = [q.strip() for q in examples[params.text_column]] # text_column is question
            contexts = examples[params.context_column]
            inputs = tokenizer(
                questions,
                contexts,
                max_length=params.max_seq_length,
                truncation="only_second", # Truncate context if too long
                stride=128, # Example stride for overlapping contexts
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            
            offset_mapping = inputs.pop("offset_mapping")
            sample_mapping = inputs.pop("overflow_to_sample_mapping")
            answers = examples[params.label_column] # label_column should point to 'answers' field
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_mapping[i]
                answer = answers[sample_idx] # SQuAD answers are dicts {'text': [...], 'answer_start': [...]}
                
                # For simplicity, assuming answer is a dict with 'answer_start' and 'text'
                # This part is complex for robust SQuAD preprocessing.
                # This is a simplified version.
                if not answer['answer_start']: # Handle cases with no answer
                    start_positions.append(0)
                    end_positions.append(0)
                    continue

                start_char = answer['answer_start'][0]
                end_char = start_char + len(answer['text'][0])
                
                sequence_ids = inputs.sequence_ids(i)

                # Find start and end token positions
                context_start_idx = sequence_ids.index(1) # 1 for context
                context_end_idx = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

                # If answer is not in the current span
                if not (offset[context_start_idx][0] <= start_char and offset[context_end_idx][1] >= end_char):
                    start_positions.append(0) # CLS token
                    end_positions.append(0)   # CLS token
                else:
                    # Find token start and end
                    token_start_index = context_start_idx
                    while token_start_index <= context_end_idx and offset[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    token_end_index = context_end_idx
                    while token_end_index >= context_start_idx and offset[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)
            
            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs

        def tokenize_fn_generic(ex):
            if params.text_column not in ex:
                raise ValueError(f"Text column '{params.text_column}' missing in tokenization input.")
            tk_batch = tokenizer(ex[params.text_column], truncation=True, padding="max_length", max_length=params.max_seq_length)
            if request.task == "text-classification":
                if params.label_column not in ex:
                    raise ValueError(f"Label column '{params.label_column}' missing for classification.")
                tk_batch["labels"] = ex[params.label_column]
            return tk_batch
        
        chosen_tokenize_fn = tokenize_fn_qa if request.task == "question-answering" else tokenize_fn_generic
        
        train_split, eval_split = "train", "validation" if "validation" in raw_datasets else "test" if "test" in raw_datasets else None
        if train_split not in raw_datasets: raise HTTPException(status_code=400, detail=f"'{train_split}' not in dataset.")
        if not eval_split and request.task != "question-answering": # QA fine-tuning might not always have a simple eval split in Trainer
             logger.warning("No validation/test split for evaluation.")

        # For QA, remove_columns needs to be handled carefully due to SQuAD format.
        # It's better to select columns explicitly or handle it after map for QA.
        cols_to_remove_for_map = raw_datasets[train_split].column_names
        if request.task == "question-answering":
            # For QA, the tokenize_fn_qa already handles selecting necessary fields and creating new ones.
            # We remove original columns that are not inputs or targets for the model.
            # The SQuAD dataset has 'id', 'title', 'context', 'question', 'answers'.
            # Our tokenize_fn_qa creates 'input_ids', 'attention_mask', 'start_positions', 'end_positions'.
            # So, we can remove the original SQuAD columns.
            cols_to_remove_for_map = [col for col in raw_datasets[train_split].column_names if col not in ['input_ids', 'attention_mask', 'start_positions', 'end_positions', 'offset_mapping', 'overflow_to_sample_mapping']]


        tokenized_ds = raw_datasets.map(chosen_tokenize_fn, batched=True, remove_columns=cols_to_remove_for_map)
        
        num_labels = params.num_labels # For classification
        if request.task == "text-classification" and not num_labels:
            if "labels" not in tokenized_ds[train_split].features:
                raise HTTPException(status_code=400, detail=f"Column 'labels' not found in tokenized training data.")
            labels_feature = tokenized_ds[train_split].features.get("labels")
            if hasattr(labels_feature, "num_classes"): num_labels = labels_feature.num_classes
            else: 
                unique_labels_list = tokenized_ds[train_split].unique("labels")
                if unique_labels_list: num_labels = len(unique_labels_list)
            if not num_labels: raise HTTPException(status_code=400, detail="num_labels needed and not inferable.")
            logger.info(f"Inferred num_labels: {num_labels}")

        base_model_args = {}
        if request.task == "text-classification": base_model_args["num_labels"] = num_labels
        
        ModelClass = None
        if request.task == "text-classification": ModelClass = AutoModelForSequenceClassification
        elif request.task == "text-generation": ModelClass = AutoModelForCausalLM
        elif request.task == "question-answering": ModelClass = AutoModelForQuestionAnswering
        
        if not ModelClass: raise HTTPException(status_code=400, detail=f"Fine-tuning for task {request.task} unsupported.")
        
        if request.task == "text-classification" and num_labels is not None:
            base_model_args["ignore_mismatched_sizes"] = True 

        base_model = ModelClass.from_pretrained(request.model_name, **base_model_args)
        
        if tokenizer.pad_token_id is not None and hasattr(base_model, 'resize_token_embeddings') and tokenizer.pad_token_id >= base_model.config.vocab_size:
             logger.info(f"Resizing token embeddings for {request.model_name} during fine-tuning setup due to new pad_token.")
             base_model.resize_token_embeddings(len(tokenizer))

        peft_model = base_model
        if params.use_lora and params.lora_config:
            lora_c = params.lora_config
            lora_targets = get_lora_target_modules(request.model_name, lora_c.target_modules)
            if not lora_targets: logger.warning("No LoRA targets; LoRA may not be effective.")
            
            task_type_peft = None
            if request.task == "text-classification": task_type_peft = TaskType.SEQ_CLS
            elif request.task == "text-generation": task_type_peft = TaskType.CAUSAL_LM
            elif request.task == "question-answering": task_type_peft = TaskType.QUESTION_ANS # PEFT TaskType for QA
            
            if not task_type_peft: raise HTTPException(status_code=400, detail=f"LoRA task type undetermined for {request.task}.")
            
            peft_config = LoraConfig(r=lora_c.r, lora_alpha=lora_c.alpha, lora_dropout=lora_c.dropout, target_modules=lora_targets, bias="none", task_type=task_type_peft)
            peft_model = get_peft_model(base_model, peft_config)
            peft_model.print_trainable_parameters()

        training_args_dict = {
            "output_dir": output_dir, "num_train_epochs": params.epochs,
            "per_device_train_batch_size": params.batch_size, "per_device_eval_batch_size": params.batch_size,
            "learning_rate": params.learning_rate, "weight_decay": 0.01,
            "logging_dir": f"{output_dir}/logs", "logging_steps": max(1, int(len(tokenized_ds[train_split]) / (params.batch_size * 5))),
            "evaluation_strategy": "epoch" if eval_split and eval_split in tokenized_ds and request.task != "question-answering" else "no", # QA eval with Trainer needs specific metrics
            "save_strategy": "epoch" if eval_split and eval_split in tokenized_ds and request.task != "question-answering" else "steps",
            "save_total_limit": 2, 
            "load_best_model_at_end": bool(eval_split and eval_split in tokenized_ds and request.task != "question-answering"),
            "no_cuda": True, "report_to": "none"
        }
        # For QA, Trainer needs a specific DataCollator if not using default padding.
        # And evaluation needs a specific compute_metrics function for SQuAD metrics.
        # This example keeps it simple and might skip eval for QA with Trainer for now.
        data_collator_for_trainer = DataCollatorWithPadding(tokenizer=tokenizer)
        if request.task == "question-answering":
            logger.info("Using default DataCollatorWithPadding for QA fine-tuning. For SQuAD metrics, custom eval is needed.")
            # For robust QA fine-tuning, you'd use DataCollatorForQuestionAnswering if needed
            # and a compute_metrics function that calculates F1/EM for SQuAD.
            # TrainingArguments might also need `predict_with_generate=False` for QA.

        training_args = TrainingArguments(**training_args_dict)
        trainer = Trainer(
            model=peft_model, 
            args=training_args, 
            train_dataset=tokenized_ds[train_split], 
            eval_dataset=tokenized_ds.get(eval_split) if request.task != "question-answering" else None, # Simpler to omit eval for QA with basic Trainer
            tokenizer=tokenizer, 
            data_collator=data_collator_for_trainer
        )
        
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
