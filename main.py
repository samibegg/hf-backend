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

# --- Pydantic Models for API (Content from previous version) ---
class InferenceRequest(BaseModel):
    task: str = Field(..., description="NLP task")
    model_name: str = Field(..., description="Hugging Face model identifier")
    input_text: Union[str, List[str]] = Field(..., description="Text to process")
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
    text_column: Optional[str] = Field("text")
    label_column: Optional[str] = Field("label")
    epochs: int = Field(3)
    batch_size: int = Field(8)
    learning_rate: float = Field(2e-5)
    use_lora: bool = Field(True)
    lora_config: Optional[LoRAConfigModel] = None
    output_dir_base: str = Field("./finetuned_models")
    max_seq_length: int = Field(256)
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

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hugging Face Model Runner API",
    description="API to run inference and fine-tune Hugging Face models.",
    version="0.1.0"
)

# --- CORS Middleware Configuration ---
# IMPORTANT: Adjust origins as needed for your environment.
# For development, you might allow "http://localhost:3000" (if your Next.js runs there).
# For production, specify your frontend's domain: "https://yourdomain.com".
# Using ["*"] allows all origins, which is convenient for development but
# should be restricted in production for security.

origins = [
    "http://localhost:3000",  # For local Next.js development
    "https://yourdomain.com", # Replace with your frontend's production domain
    # Add other origins if needed, e.g., a staging environment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins that are allowed to make requests
    allow_credentials=True, # Allow cookies to be included in requests
    allow_methods=["*"],    # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],    # Allow all headers
)

# --- Global Variables / Model Cache (Content from previous version) ---
loaded_models_cache = {}

# --- Helper Functions (Content from previous version, no changes needed here for CORS) ---
def get_model_and_tokenizer(model_name: str, task: str, num_labels: Optional[int] = None, quantization: str = "none"):
    """Loads model and tokenizer, applying quantization if specified. Caches them."""
    # For text-classification, the num_labels in the cache key is important if different heads are used.
    # For inference, num_labels is usually None, and for fine-tuning it's specified.
    cache_key = (model_name, quantization, task, num_labels if task == "text-classification" else None)
    if cache_key in loaded_models_cache:
        logger.info(f"Using cached model: {model_name} ({quantization}, task: {task}, num_labels for cache key: {cache_key[3]})")
        return loaded_models_cache[cache_key]

    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Loading model {model_name} for task {task}...")

    # Set pad_token if not present, common for generative models
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.info(f"Tokenizer for {model_name} has no pad_token. Setting pad_token to eos_token ({tokenizer.eos_token}).")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # If no EOS token either, add a new pad token. This is less common for pre-trained.
            # For models like GPT2, eos_token is usually also bos_token and pad_token.
            # If truly no eos_token, adding a new special token might be necessary,
            # but could affect model performance if not trained with it.
            # Defaulting to a common strategy for models that might lack it.
            logger.warning(f"Tokenizer for {model_name} has no pad_token and no eos_token. Adding a new pad_token '[PAD]'. This might require model retraining for optimal performance if the model was not expecting it.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # If you add a new token, the model's embedding layer might need resizing if you intend to fine-tune.
            # For inference, this might be okay if the model isn't sensitive to it or if padding is on the right.

    if task == "text-classification":
        if num_labels is not None:
            # This branch is typically for fine-tuning, where num_labels is explicitly set.
            logger.info(f"Loading {model_name} for text-classification with explicit num_labels: {num_labels}")
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        else:
            # This branch is for inference. Let the model load its own config for num_labels.
            logger.info(f"Loading {model_name} for text-classification inference (num_labels will be inferred from model config).")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task == "text-generation":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # If tokenizer vocabulary was expanded by adding a new pad_token, resize model embeddings
        # This is more critical for fine-tuning, but good practice if vocab changed.
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= model.config.vocab_size:
             logger.info(f"Resizing token embeddings for {model_name} due to new pad_token.")
             model.resize_token_embeddings(len(tokenizer))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task for model loading: {task}")
    
    model.eval() # Set to evaluation mode by default

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
    if "distilbert" in name: return ["q_lin", "k_lin", "v_lin", "out_lin"]
    if "roberta" in name or "bert" in name: return ["query", "key", "value", "dense"]
    if any(n in name for n in ["gpt2", "llama", "mistral", "opt"]):
        logger.warning(f"Generic LoRA targets for {name}. Verify for optimal targets.")
        return ["c_attn", "c_proj"] # GPT-2 like, adjust for others (e.g. q_proj, v_proj for Llama)
    logger.warning(f"Could not infer LoRA targets for {name}.")
    return []

# --- API Endpoints (Content from previous version, no changes needed here for CORS) ---
@app.post("/api/v1/infer", response_model=InferenceResponse)
async def run_inference_endpoint(request: InferenceRequest):
    logger.info(f"Received inference request: Task={request.task}, Model={request.model_name}")
    try:
        # For inference, num_labels is not passed to get_model_and_tokenizer, so it defaults to None.
        # The updated get_model_and_tokenizer will handle this correctly for text-classification.
        model, tokenizer = get_model_and_tokenizer(
            model_name=request.model_name,
            task=request.task,
            quantization=request.quantization
        )

        if request.task == "text-classification":
            # Using pipeline for simplicity for classification
            # The pipeline will use the model's loaded configuration (including num_labels).
            classifier = pipeline(
                "sentiment-analysis" if "sentiment" in request.model_name.lower() or "sst-2" in request.model_name.lower() else "text-classification", # Improved heuristic
                model=model,
                tokenizer=tokenizer,
                device=-1 # -1 for CPU
            )
            results = classifier(request.input_text if isinstance(request.input_text, list) else [request.input_text])
            predictions = results

        elif request.task == "text-generation":
            # Ensure tokenizer has pad_token_id for padding=True
            if tokenizer.pad_token_id is None:
                 # This should have been handled in get_model_and_tokenizer, but as a safeguard:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    # This case should be very rare for pre-trained models
                    raise ValueError("Tokenizer has no pad_token_id and no eos_token_id. Cannot proceed with padding for generation.")
            
            inputs = tokenizer(request.input_text, return_tensors="pt", padding=True, truncation=True)
            gen_args = request.generation_args or {}
            final_gen_args = {**{"max_length": 50, "num_return_sequences": 1}, **gen_args} # Default generation args
            with torch.no_grad(): outputs = model.generate(**inputs, **final_gen_args)
            
            # Decode all generated sequences if input_text was a list, else decode the first one
            if isinstance(request.input_text, list) and len(outputs) == len(inputs["input_ids"]) * final_gen_args.get("num_return_sequences", 1) :
                 # This logic might need adjustment based on how batch generation returns sequences
                predictions = [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in outputs]
            elif not isinstance(request.input_text, list):
                predictions = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else: # Fallback for batched input if logic above is not perfect
                logger.warning("Decoding multiple sequences for batched text generation, structure might vary.")
                predictions = [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in outputs]


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
async def run_fine_tuning_endpoint(request: FineTuneRequest, background_tasks: BackgroundTasks): # Renamed
    params = request.fine_tune_params
    logger.info(f"Fine-tune: Task={request.task}, Model={request.model_name}, Data={params.dataset_path}")
    run_name = f"{request.model_name.replace('/', '_')}_{request.task}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(params.output_dir_base, run_name)
    os.makedirs(output_dir, exist_ok=True)
    adapter_output_path = os.path.join(output_dir, "lora_adapters")

    try:
        tokenizer = AutoTokenizer.from_pretrained(request.model_name)
        # Pad token handling for fine-tuning tokenizer (also done in get_model_and_tokenizer, but good to ensure here too)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Fine-tuning: Set pad_token to eos_token ({tokenizer.eos_token}) for {request.model_name}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.warning(f"Fine-tuning: Added new pad_token '[PAD]' for {request.model_name}")


        if params.dataset_path.endswith(".csv"):
            df = pd.read_csv(params.dataset_path)
            # Ensure text and label columns exist before trying to use them
            if params.text_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Text column '{params.text_column}' not found in CSV.")
            if request.task == "text-classification" and params.label_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Label column '{params.label_column}' not found in CSV for text classification.")

            train_df = df.sample(frac=0.8, random_state=42)
            eval_df = df.drop(train_df.index)
            raw_datasets = DatasetDict({"train": Dataset.from_pandas(train_df), "validation": Dataset.from_pandas(eval_df)})
        else:
            raw_datasets = load_dataset(params.dataset_path)

        def tokenize_fn(ex):
            # Check if text_column exists in the current example batch (ex)
            if params.text_column not in ex:
                logger.error(f"Text column '{params.text_column}' not found in example batch during tokenization. Keys: {list(ex.keys())}")
                raise ValueError(f"Text column '{params.text_column}' missing in tokenization input.")

            tk_batch = tokenizer(ex[params.text_column], truncation=True, padding="max_length", max_length=params.max_seq_length)
            
            if request.task == "text-classification":
                if params.label_column not in ex:
                    logger.error(f"Label column '{params.label_column}' not found in example batch for text classification. Keys: {list(ex.keys())}")
                    raise ValueError(f"Label column '{params.label_column}' missing in tokenization input for classification.")
                tk_batch["labels"] = ex[params.label_column]
            return tk_batch
        
        train_split, eval_split = "train", "validation" if "validation" in raw_datasets else "test" if "test" in raw_datasets else None
        if train_split not in raw_datasets: raise HTTPException(status_code=400, detail=f"'{train_split}' not in dataset.")
        if not eval_split: logger.warning("No validation/test split for evaluation.")

        tokenized_ds = raw_datasets.map(tokenize_fn, batched=True, remove_columns=raw_datasets[train_split].column_names) # Remove all original columns
        
        num_labels = params.num_labels
        if request.task == "text-classification" and not num_labels:
            # Ensure 'labels' column exists after tokenization before trying to access its features
            if "labels" not in tokenized_ds[train_split].features:
                raise HTTPException(status_code=400, detail=f"Column 'labels' not found in tokenized training data. Check label_column ('{params.label_column}') and tokenization.")

            labels_feature = tokenized_ds[train_split].features.get("labels")
            if hasattr(labels_feature, "num_classes"): num_labels = labels_feature.num_classes
            else: # Fallback: count unique labels if it's not a ClassLabel feature
                unique_labels_list = tokenized_ds[train_split].unique("labels")
                if unique_labels_list:
                    num_labels = len(unique_labels_list)
            if not num_labels: raise HTTPException(status_code=400, detail="num_labels needed and not inferable.")
            logger.info(f"Inferred num_labels: {num_labels}")

        base_model_args = {"num_labels": num_labels} if request.task == "text-classification" else {}
        ModelClass = AutoModelForSequenceClassification if request.task == "text-classification" else AutoModelForCausalLM if request.task == "text-generation" else None
        if not ModelClass: raise HTTPException(status_code=400, detail=f"Fine-tuning for task {request.task} unsupported.")
        
        if request.task == "text-classification" and num_labels is not None:
            base_model_args["ignore_mismatched_sizes"] = True 

        base_model = ModelClass.from_pretrained(request.model_name, **base_model_args)
        
        # If tokenizer vocabulary was expanded by adding a new pad_token, resize model embeddings
        # This is important before PEFT or full fine-tuning.
        if tokenizer.pad_token_id is not None and hasattr(base_model, 'resize_token_embeddings') and tokenizer.pad_token_id >= base_model.config.vocab_size:
             logger.info(f"Resizing token embeddings for {request.model_name} during fine-tuning setup due to new pad_token.")
             base_model.resize_token_embeddings(len(tokenizer))

        peft_model = base_model
        if params.use_lora and params.lora_config:
            lora_c = params.lora_config
            lora_targets = get_lora_target_modules(request.model_name, lora_c.target_modules)
            if not lora_targets: logger.warning("No LoRA targets; LoRA may not be effective.")
            
            task_type_peft = TaskType.SEQ_CLS if request.task == "text-classification" else TaskType.CAUSAL_LM if request.task == "text-generation" else None
            if not task_type_peft: raise HTTPException(status_code=400, detail=f"LoRA task type undetermined for {request.task}.")
            
            peft_config = LoraConfig(r=lora_c.r, lora_alpha=lora_c.alpha, lora_dropout=lora_c.dropout, target_modules=lora_targets, bias="none", task_type=task_type_peft)
            peft_model = get_peft_model(base_model, peft_config)
            peft_model.print_trainable_parameters()

        training_args_dict = {
            "output_dir": output_dir, "num_train_epochs": params.epochs,
            "per_device_train_batch_size": params.batch_size, "per_device_eval_batch_size": params.batch_size,
            "learning_rate": params.learning_rate, "weight_decay": 0.01,
            "logging_dir": f"{output_dir}/logs", "logging_steps": max(1, int(len(tokenized_ds[train_split]) / (params.batch_size * 5))),
            "evaluation_strategy": "epoch" if eval_split and eval_split in tokenized_ds else "no",
            "save_strategy": "epoch" if eval_split and eval_split in tokenized_ds else "steps",
            "save_total_limit": 2, "load_best_model_at_end": bool(eval_split and eval_split in tokenized_ds),
            "no_cuda": True, "report_to": "none"
        }
        training_args = TrainingArguments(**training_args_dict)
        trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized_ds[train_split], eval_dataset=tokenized_ds.get(eval_split), tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer))
        
        logger.info("Starting fine-tuning...")
        trainer.train()
        logger.info("Fine-tuning finished.")
        
        final_save_path = adapter_output_path if params.use_lora else os.path.join(output_dir, "full_model_final")
        if params.use_lora: peft_model.save_pretrained(final_save_path)
        else: trainer.save_model(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        
        for item in os.listdir(output_dir): # Cleanup checkpoints
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
