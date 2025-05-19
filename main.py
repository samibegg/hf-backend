# main.py
# Main FastAPI application file

import os
import shutil
import logging
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    # Add other AutoModel types as needed, e.g.:
    # AutoModelForQuestionAnswering,
    # AutoModelForTokenClassification,
    # AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline # For simpler inference on some tasks
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import numpy as np
import pandas as pd # For loading CSV datasets

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure model/dataset cache directory exists and is writable if needed
# Hugging Face defaults to ~/.cache/huggingface/
# You can set TRANSFORMERS_CACHE or HF_HOME environment variables

# --- Pydantic Models for API ---

class InferenceRequest(BaseModel):
    task: str = Field(..., description="NLP task, e.g., 'text-classification', 'text-generation'")
    model_name: str = Field(..., description="Hugging Face model identifier")
    input_text: Union[str, List[str]] = Field(..., description="Text to process or a list of texts")
    quantization: Optional[str] = Field("none", description="Quantization option: 'none', 'dynamic_int8_cpu'")
    generation_args: Optional[Dict[str, Any]] = Field(None, description="Arguments for text generation (e.g., max_length, num_beams)")

class InferenceResponse(BaseModel):
    predictions: Any # Could be a list of dicts, a string, etc.
    model_used: str
    quantization_applied: str

class LoRAConfigModel(BaseModel):
    r: int = Field(8, description="Rank of LoRA decomposition")
    alpha: int = Field(16, description="LoRA scaling factor")
    dropout: float = Field(0.05, description="Dropout for LoRA layers")
    target_modules: Optional[Union[str, List[str]]] = Field("q_proj,v_proj", description="Comma-separated string or list of target module names for LoRA")

class FineTuneParamsModel(BaseModel):
    dataset_path: str = Field(..., description="Path or HF Hub name of the dataset")
    text_column: Optional[str] = Field("text", description="Name of the text column in the dataset")
    label_column: Optional[str] = Field("label", description="Name of the label column (for classification)")
    epochs: int = Field(3, description="Number of training epochs")
    batch_size: int = Field(8, description="Training batch size")
    learning_rate: float = Field(2e-5, description="Learning rate")
    use_lora: bool = Field(True, description="Whether to use LoRA for fine-tuning")
    lora_config: Optional[LoRAConfigModel] = None
    output_dir_base: str = Field("./finetuned_models", description="Base directory to save fine-tuned models")
    max_seq_length: int = Field(256, description="Maximum sequence length for tokenization")
    num_labels: Optional[int] = Field(None, description="Number of labels for classification (if not inferable)")


class FineTuneRequest(BaseModel):
    task: str = Field(..., description="NLP task for fine-tuning")
    model_name: str = Field(..., description="Base Hugging Face model identifier")
    fine_tune_params: FineTuneParamsModel

class FineTuneResponse(BaseModel):
    message: str
    model_name: str
    adapter_output_path: Optional[str] = None
    logs: Optional[str] = None # Could be a path to a log file or inline logs

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Hugging Face Model Runner API",
    description="API to run inference and fine-tune Hugging Face models.",
    version="0.1.0"
)

# --- Global Variables / Model Cache (Simple In-Memory) ---
# For a production app, consider a more robust caching mechanism or loading models on demand.
# This simple cache might not be ideal for very large models or many concurrent users.
loaded_models_cache = {} # Stores { (model_name, quantization_type, task_type_for_head): (model, tokenizer) }

# --- Helper Functions ---

def get_model_and_tokenizer(model_name: str, task: str, num_labels: Optional[int] = None, quantization: str = "none"):
    """Loads model and tokenizer, applying quantization if specified. Caches them."""
    cache_key = (model_name, quantization, task, num_labels) # num_labels part of key for classification heads
    if cache_key in loaded_models_cache:
        logger.info(f"Using cached model and tokenizer for {model_name} ({quantization}, task: {task})")
        return loaded_models_cache[cache_key]

    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loading model {model_name} for task {task}...")
    if task == "text-classification":
        if num_labels is None:
            # Try to infer from config, or raise error if not possible for fine-tuning
            # For pre-finetuned models, this is usually set.
            # config = AutoConfig.from_pretrained(model_name)
            # num_labels = config.num_labels if hasattr(config, 'num_labels') else 2 # Default or raise
            logger.warning(f"num_labels not explicitly provided for {model_name} text-classification. It might be inferred if model is already fine-tuned for classification.")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    elif task == "text-generation":
        model = AutoModelForCausalLM.from_pretrained(model_name)
    # Add other task-specific model loaders here
    # elif task == "question-answering":
    #     model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task for model loading: {task}")

    model.eval() # Set to evaluation mode by default

    if quantization == "dynamic_int8_cpu":
        if not hasattr(torch.quantization, 'quantize_dynamic'):
             raise HTTPException(status_code=501, detail="Dynamic quantization not available in this PyTorch build.")
        logger.info(f"Applying dynamic INT8 quantization to {model_name} for CPU...")
        # Ensure model is on CPU before quantization if it was loaded on GPU by mistake
        model = model.to("cpu")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info("Dynamic quantization applied.")
    elif quantization != "none":
        logger.warning(f"Unsupported quantization option '{quantization}' for on-the-fly loading. Using 'none'.")


    loaded_models_cache[cache_key] = (model, tokenizer)
    return model, tokenizer

def get_lora_target_modules(model_name_or_path: str, user_specified_modules: Optional[Union[str, List[str]]]) -> List[str]:
    """
    Determines LoRA target modules.
    If user_specified_modules is provided (as comma-separated string or list), use that.
    Otherwise, provide some common defaults based on model type.
    This is a simplified heuristic; for best results, users should specify based on model architecture.
    """
    if isinstance(user_specified_modules, str):
        modules = [m.strip() for m in user_specified_modules.split(',') if m.strip()]
        if modules:
            logger.info(f"Using user-specified LoRA target modules: {modules}")
            return modules
    elif isinstance(user_specified_modules, list) and user_specified_modules:
        logger.info(f"Using user-specified LoRA target modules: {user_specified_modules}")
        return user_specified_modules

    logger.info("User did not specify LoRA target modules. Attempting to use common defaults.")
    model_name_lower = model_name_or_path.lower()
    if "distilbert" in model_name_lower:
        return ["q_lin", "k_lin", "v_lin", "out_lin"] # "ffn.lin1", "ffn.lin2" also common
    elif "roberta" in model_name_lower or "bert" in model_name_lower : # and not "distilbert"
        return ["query", "key", "value", "dense"] # "intermediate.dense", "output.dense" for FFN
    elif "gpt2" in model_name_lower or "llama" in model_name_lower or "mistral" in model_name_lower or "opt" in model_name_lower:
        # Common for many decoder models, but can vary (e.g., Llama uses q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
        # This is a very broad guess.
        logger.warning(f"Using generic LoRA targets for {model_name_or_path}. Please verify model architecture for optimal targets (e.g., q_proj, v_proj).")
        return ["c_attn", "c_proj"] # GPT-2 like
    else:
        logger.warning(f"Could not determine default LoRA target modules for {model_name_or_path}. LoRA might not be applied effectively without explicit target_modules.")
        return []


# --- API Endpoints ---

@app.post("/api/v1/infer", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    logger.info(f"Received inference request: Task={request.task}, Model={request.model_name}")
    try:
        # For text classification, num_labels might be needed if the model isn't pre-fine-tuned
        # However, for inference, the head should already exist.
        model, tokenizer = get_model_and_tokenizer(request.model_name, request.task, quantization=request.quantization)

        if request.task == "text-classification":
            # Using pipeline for simplicity for classification
            classifier = pipeline(
                "sentiment-analysis" if "sentiment" in request.model_name else "text-classification", # Heuristic for pipeline task
                model=model,
                tokenizer=tokenizer,
                device=-1 # -1 for CPU, 0 for GPU 0 (if available and model on GPU)
            )
            # Pipeline expects a list of strings or a single string
            results = classifier(request.input_text if isinstance(request.input_text, list) else [request.input_text])
            predictions = results

        elif request.task == "text-generation":
            inputs = tokenizer(request.input_text, return_tensors="pt", padding=True, truncation=True)
            # Ensure inputs are on the same device as the model (CPU in this case)
            # inputs = {k: v.to(model.device) for k, v in inputs.items()}

            gen_args = request.generation_args or {}
            default_gen_args = {"max_length": 50, "num_return_sequences": 1}
            final_gen_args = {**default_gen_args, **gen_args}

            with torch.no_grad():
                outputs = model.generate(**inputs, **final_gen_args)
            
            if isinstance(request.input_text, list):
                predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            else:
                predictions = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
async def run_fine_tuning(request: FineTuneRequest, background_tasks: BackgroundTasks):
    """
    Fine-tunes a model.
    NOTE: This is a SYNCHRONOUS endpoint for simplicity. For production,
    long-running tasks like fine-tuning should be handled asynchronously
    using background_tasks properly (e.g., returning a job ID and having
    separate status/result endpoints) or a dedicated task queue like Celery.
    The current BackgroundTasks usage is a placeholder for where that logic would go.
    """
    params = request.fine_tune_params
    logger.info(f"Received fine-tuning request: Task={request.task}, Model={request.model_name}, Dataset={params.dataset_path}")

    # Create a unique output directory for this run
    run_name = f"{request.model_name.replace('/', '_')}_{request.task}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(params.output_dir_base, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    adapter_output_path = os.path.join(output_dir, "lora_adapters")

    try:
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(request.model_name)
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token


        # 2. Load and Preprocess Dataset
        logger.info(f"Loading dataset: {params.dataset_path}")
        # Simplistic dataset loading: assumes Hugging Face dataset or local CSV
        if params.dataset_path.endswith(".csv"):
            # For CSV, expect 'train.csv' and 'validation.csv' or handle splits
            # This is a simplified example assuming one CSV for train/eval split
            try:
                df = pd.read_csv(params.dataset_path)
                # Simple split for demo, replace with proper train/test/validation split
                train_df = df.sample(frac=0.8, random_state=42)
                eval_df = df.drop(train_df.index)
                
                raw_datasets = Dataset.from_pandas(train_df).train_test_split(test_size=eval_df.shape[0]/df.shape[0], seed=42)
                # This is not ideal, better to have separate train/eval files or a split column
                # For now, let's assume 'train' and 'test' (for validation) splits exist after this
                # Or, if you load a single CSV, you might just use it all for training and a small part for eval.
                # For simplicity, assuming the split above works or user provides HF dataset with splits
                # Example: raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'eval': 'eval.csv'})
            except Exception as e:
                 raise HTTPException(status_code=400, detail=f"Failed to load or process CSV dataset {params.dataset_path}: {e}")
        else: # Assume Hugging Face dataset identifier
            raw_datasets = load_dataset(params.dataset_path)

        def tokenize_function(examples):
            # Ensure text_column and label_column are correctly used
            tokenized_batch = tokenizer(
                examples[params.text_column],
                truncation=True,
                padding="max_length", # Or handle by data collator
                max_length=params.max_seq_length
            )
            if params.label_column in examples: # For classification
                 tokenized_batch["labels"] = examples[params.label_column]
            return tokenized_batch

        logger.info("Tokenizing dataset...")
        # Ensure required columns exist
        required_cols_train = [params.text_column]
        if request.task == "text-classification": required_cols_train.append(params.label_column)
        
        for col in required_cols_train:
            if col not in raw_datasets["train"].column_names:
                raise HTTPException(status_code=400, detail=f"Required column '{col}' not found in training dataset.")

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        
        # Remove columns not needed by the model
        columns_to_remove = [col for col in raw_datasets["train"].column_names if col not in [params.text_column, params.label_column, "input_ids", "attention_mask", "labels", "token_type_ids"]]
        if params.text_column not in ["input_ids", "attention_mask", "labels"]: columns_to_remove.append(params.text_column)
        if params.label_column not in ["input_ids", "attention_mask", "labels"]: columns_to_remove.append(params.label_column)
        
        tokenized_datasets = tokenized_datasets.remove_columns(list(set(columns_to_remove) & set(tokenized_datasets["train"].column_names)))
        
        if request.task == "text-classification" and "label" in tokenized_datasets["train"].column_names and "labels" not in tokenized_datasets["train"].column_names:
             tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


        # 3. Load Base Model
        logger.info(f"Loading base model: {request.model_name}")
        num_labels_for_task = params.num_labels
        if request.task == "text-classification" and not num_labels_for_task:
            # Try to infer from dataset if possible
            if 'labels' in tokenized_datasets["train"].features:
                num_labels_for_task = tokenized_datasets["train"].features['labels'].num_classes
                logger.info(f"Inferred num_labels from dataset: {num_labels_for_task}")
            else:
                raise HTTPException(status_code=400, detail="num_labels is required for text-classification and could not be inferred from the dataset.")

        if request.task == "text-classification":
            base_model = AutoModelForSequenceClassification.from_pretrained(request.model_name, num_labels=num_labels_for_task)
        elif request.task == "text-generation":
            base_model = AutoModelForCausalLM.from_pretrained(request.model_name)
        else:
            raise HTTPException(status_code=400, detail=f"Fine-tuning for task '{request.task}' not supported yet.")

        # 4. PEFT/LoRA Configuration (if enabled)
        peft_model = base_model
        if params.use_lora and params.lora_config:
            logger.info("Configuring LoRA...")
            lora_c = params.lora_config
            
            # Determine target modules
            lora_target_modules_list = get_lora_target_modules(request.model_name, lora_c.target_modules)
            if not lora_target_modules_list:
                 logger.warning("No LoRA target modules specified or inferred. LoRA may not be effective.")


            peft_lora_config = LoraConfig(
                r=lora_c.r,
                lora_alpha=lora_c.alpha,
                lora_dropout=lora_c.dropout,
                target_modules=lora_target_modules_list,
                bias="none",
                task_type=TaskType.SEQ_CLS if request.task == "text-classification" else \
                          TaskType.CAUSAL_LM if request.task == "text-generation" else None # Add other task types
            )
            if peft_lora_config.task_type is None:
                raise HTTPException(status_code=400, detail=f"LoRA task type could not be determined for task: {request.task}")

            peft_model = get_peft_model(base_model, peft_lora_config)
            peft_model.print_trainable_parameters()
        else:
            logger.info("LoRA not enabled or lora_config not provided. Performing full fine-tuning (if applicable).")


        # 5. Training Arguments and Trainer
        logger.info("Setting up Training Arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=params.epochs,
            per_device_train_batch_size=params.batch_size,
            per_device_eval_batch_size=params.batch_size, # Can be different
            learning_rate=params.learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100, # Log every 100 steps
            evaluation_strategy="epoch", # Evaluate at the end of each epoch
            save_strategy="epoch",       # Save model at the end of each epoch
            load_best_model_at_end=True, # Load the best model found during training
            # For CPU-only VM:
            no_cuda=True,
            # use_cpu=True, # Not a standard TrainingArgument, accelerate handles this
            report_to="none" # Disable wandb/tensorboard reporting for simplicity
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation") or tokenized_datasets.get("test"), # Use validation or test
            tokenizer=tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_metrics_function, # Optional: for more detailed eval
        )

        # 6. Start Training
        logger.info("Starting fine-tuning...")
        train_result = trainer.train()
        logger.info("Fine-tuning finished.")
        # trainer.log_metrics("train", train_result.metrics) # Log training metrics
        # trainer.save_metrics("train", train_result.metrics)

        # 7. Save Model/Adapters
        if params.use_lora:
            logger.info(f"Saving LoRA adapters to {adapter_output_path}")
            peft_model.save_pretrained(adapter_output_path)
            tokenizer.save_pretrained(adapter_output_path) # Save tokenizer with adapters
            final_save_path = adapter_output_path
        else: # Full fine-tuning
            full_model_path = os.path.join(output_dir, "full_model")
            logger.info(f"Saving full fine-tuned model to {full_model_path}")
            trainer.save_model(full_model_path) # Saves the full model
            tokenizer.save_pretrained(full_model_path)
            final_save_path = full_model_path
        
        # Clean up checkpoint directories within output_dir to save space, keep only the best
        # (Trainer with load_best_model_at_end=True might handle some of this,
        # but explicit cleanup of intermediate checkpoints can be useful)
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                # Check if it's not the final saved model path (if using save_model directly)
                # or if it's not the adapter path
                if final_save_path != item_path and not final_save_path.startswith(item_path):
                    logger.info(f"Removing intermediate checkpoint: {item_path}")
                    shutil.rmtree(item_path)

        return FineTuneResponse(
            message="Fine-tuning completed successfully.",
            model_name=request.model_name,
            adapter_output_path=final_save_path if params.use_lora else None, # or model_output_path for full
            logs=f"Training logs and metrics saved in {output_dir}"
        )

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}", exc_info=True)
        # Clean up failed run directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Welcome to the Hugging Face Model Runner API!"}

# To run this app (save as main.py):
# 1. Install dependencies: pip install fastapi uvicorn python-multipart torch transformers datasets peft pandas sentencepiece accelerate bitsandbytes
#    For CPU PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu
# 2. Run with Uvicorn: uvicorn main:app --reload --host 0.0.0.0 --port 8000
