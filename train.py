"""
Training script for finetuning LLMs on RE-DocRED for relation extraction.
Uses HuggingFace Transformers with Trainer API.
"""
import json
import logging
import os
import torch
import shutil
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Fix tokenizer parallelism warning and memory optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

from config import ModelConfig, TrainingConfig, DataConfig
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import QLoRA dependencies, fall back if not available
try:
    from transformers import BitsAndBytesConfig
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training
    )
    QLORA_AVAILABLE = True
    logger.info("✅ QLoRA dependencies available")
except ImportError as e:
    QLORA_AVAILABLE = False
    logger.warning(f"⚠️ QLoRA not available: {e}")
    logger.warning("Will use standard fine-tuning instead")


@dataclass
class RelationExtractionDataset:
    """dataset class for relation extraction."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # label mapping
        unique_relations = list(set(ex['relation'] for ex in examples))
        self.label2id = {label: i for i, label in enumerate(sorted(unique_relations))}
        self.id2label = {i: label for label, i in self.label2id.items()}

        logger.info(f"Created dataset with {len(examples)} examples and {len(unique_relations)} relation types")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Create input text in format: "[CLS] text [SEP] head_entity [SEP] tail_entity [SEP]"
        text = example['text']
        head_entity = example['head_entity']
        tail_entity = example['tail_entity']

        # Format input for relation classification
        input_text = f"{text} [SEP] {head_entity} [SEP] {tail_entity}"

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Get label
        label = self.label2id[example['relation']]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_processed_data(data_file: str) -> List[Dict]:
    """Load preprocessed data from JSON file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data['examples']


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def save_best_model_to_drive(trainer, tokenizer, training_config, train_dataset, eval_results, use_qlora=False):
    """Save only the BEST model directly to Google Drive - no huge checkpoints!"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Try to mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        drive_available = True
        drive_path = f"/content/drive/MyDrive/RE_DocRED_Models/best_model_{timestamp}"
    except:
        logger.warning("Google Drive not available, saving locally only")
        drive_available = False
        drive_path = None

    # Save the BEST model only (not checkpoints)
    logger.info(f"Saving BEST model to {training_config.output_dir}")

    if use_qlora:
        # For QLoRA, save only the adapter weights (much smaller!)
        logger.info("💾 Saving QLoRA adapter weights (very small!)...")
        trainer.model.save_pretrained(training_config.output_dir)

        # Also save base model info for loading later
        with open(Path(training_config.output_dir) / "adapter_config.json", 'w') as f:
            json.dump({
                "base_model_name": trainer.model.peft_config['default'].base_model_name_or_path,
                "model_type": "qlora_adapter"
            }, f, indent=2)
        logger.info("✅ QLoRA adapter saved - only ~20MB instead of 13GB!")
    else:
        # Standard model saving
        trainer.save_model(training_config.output_dir)

    tokenizer.save_pretrained(training_config.output_dir)

    # Create metadata
    metadata = {
        'timestamp': timestamp,
        'model_name': training_config.output_dir,
        'training_config': {
            'batch_size': training_config.batch_size,
            'learning_rate': training_config.learning_rate,
            'num_epochs': training_config.num_epochs,
        },
        'dataset_info': {
            'num_labels': len(train_dataset.label2id),
            'label_mapping': train_dataset.label2id
        },
        'eval_results': eval_results
    }

    # Save metadata locally
    with open(Path(training_config.output_dir) / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(Path(training_config.output_dir) / "label_mapping.json", 'w') as f:
        json.dump({
            'label2id': train_dataset.label2id,
            'id2label': train_dataset.id2label
        }, f, indent=2)

    # Copy BEST model to Google Drive
    if drive_available:
        try:
            Path(drive_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying best model to Google Drive: {drive_path}")

            # Copy only essential files (not huge checkpoints)
            essential_files = [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "training_metadata.json",
                "label_mapping.json"
            ]

            for file_name in essential_files:
                src_file = Path(training_config.output_dir) / file_name
                if src_file.exists():
                    shutil.copy(src_file, Path(drive_path) / file_name)
                    logger.info(f"✅ Copied {file_name} to Drive")

            # Copy any vocab files
            for vocab_file in Path(training_config.output_dir).glob("vocab*"):
                shutil.copy(vocab_file, Path(drive_path) / vocab_file.name)

            drive_success = True
        except Exception as e:
            logger.error(f"Failed to copy to Google Drive: {e}")
            drive_success = False
    else:
        drive_success = False

    # Create summary
    summary = f"""
🎯 Model Training Complete!
========================

📅 Timestamp: {timestamp}
📁 Local Location: {training_config.output_dir}
{'💾 Google Drive: ' + drive_path if drive_success else '❌ Google Drive: Failed'}

📊 Results:
- Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}
- F1 Score: {eval_results.get('eval_f1', 'N/A'):.4f}
- Precision: {eval_results.get('eval_precision', 'N/A'):.4f}
- Recall: {eval_results.get('eval_recall', 'N/A'):.4f}

🏷️ Dataset Info:
- Number of relation types: {len(train_dataset.label2id)}
- Total training examples: {len(train_dataset.examples)}

💡 To use this model:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('{drive_path if drive_success else training_config.output_dir}')
model = AutoModelForSequenceClassification.from_pretrained('{drive_path if drive_success else training_config.output_dir}')

✨ Only BEST model saved - no huge checkpoint files!
"""

    with open("./training_summary.txt", 'w') as f:
        f.write(summary)

    print(summary)
    logger.info(f"Training summary saved to ./training_summary.txt")

    return drive_path if drive_success else training_config.output_dir


def train_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig
):
    """Main training function."""

    # Load data
    logger.info("Loading training data...")
    train_examples = load_processed_data(Path(data_config.data_dir) / "train_processed.json")
    dev_examples = load_processed_data(Path(data_config.data_dir) / "dev_processed.json")

    # Limit examples for testing if specified
    if data_config.max_examples:
        train_examples = train_examples[:data_config.max_examples]
        dev_examples = dev_examples[:min(data_config.max_examples // 4, len(dev_examples))]

    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(dev_examples)}")

    # Load tokenizer and model
    logger.info(f"Loading model: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    # Fix padding token for Mistral and other models (with debugging)
    logger.info(f"Before fix - pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            logger.info(f"Set pad_token to unk_token: {tokenizer.pad_token}")
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        else:
            # Add a new pad token if none exists
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added new [PAD] token")

    # Ensure pad_token_id is set correctly
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        logger.info(f"Set pad_token_id to: {tokenizer.pad_token_id}")

    # Final verification
    logger.info(f"After fix - pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")

    # For Mistral specifically, force EOS as pad token
    if 'mistral' in model_config.model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Mistral detected - forced pad_token to EOS: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # Create unified label mapping from ALL examples (train + dev)
    logger.info("Creating unified label mapping from train and dev data...")
    all_examples = train_examples + dev_examples
    all_relations = list(set(ex['relation'] for ex in all_examples))
    unified_label2id = {label: i for i, label in enumerate(sorted(all_relations))}
    unified_id2label = {i: label for label, i in unified_label2id.items()}

    logger.info(f"Found {len(all_relations)} unique relations across train and dev sets")

    # Create datasets with unified mapping
    train_dataset = RelationExtractionDataset(train_examples, tokenizer, model_config.max_length)
    dev_dataset = RelationExtractionDataset(dev_examples, tokenizer, model_config.max_length)

    # Override with unified mapping
    train_dataset.label2id = unified_label2id
    train_dataset.id2label = unified_id2label
    dev_dataset.label2id = unified_label2id
    dev_dataset.id2label = unified_id2label

    logger.info("✅ Both datasets now use the same unified label mapping")

    # Load model with QLoRA optimizations
    num_labels = len(train_dataset.label2id)

    # QLoRA Configuration for massive memory savings
    use_qlora = QLORA_AVAILABLE and ('mistral' in model_config.model_name.lower() or '7b' in model_config.model_name.lower())

    if use_qlora:
        logger.info("🚀 Using QLoRA for extreme memory efficiency!")

        # Ultra-aggressive 4-bit quantization config for Mistral
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,  # CPU offload for extreme memory savings
        )

        # Ultra-aggressive device mapping for Mistral
        if 'mistral' in model_config.model_name.lower():
            device_map = {
                "model.embed_tokens": "cpu",
                "model.layers.0": "cpu",
                "model.layers.1": "cpu",
                "model.layers.2": "cpu",
                "lm_head": "cpu",
                "model.norm": "cpu",
            }
            logger.info("🔄 Using CPU offload for embeddings and some layers")
        else:
            device_map = 'auto'

        model_kwargs = {
            'num_labels': num_labels,
            'quantization_config': bnb_config,
            'device_map': device_map,
            'pad_token_id': tokenizer.pad_token_id,
            'low_cpu_mem_usage': True,  # Reduce CPU memory during loading
        }
    else:
        logger.info("🔧 Loading smaller model with standard optimizations...")
        model_kwargs = {
            'num_labels': num_labels,
            'id2label': train_dataset.id2label,
            'label2id': train_dataset.label2id,
            'pad_token_id': tokenizer.pad_token_id,
            'torch_dtype': torch.float16,
            'device_map': 'auto',
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name,
        **model_kwargs
    )

    # Setup LoRA for large models
    if use_qlora:
        logger.info("⚡ Setting up LoRA adapters...")

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Ultra-efficient LoRA configuration for Mistral 7B
        if 'mistral' in model_config.model_name.lower():
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,  # Smaller rank for less memory (was 16)
                lora_alpha=16,  # Proportional scaling (was 32)
                lora_dropout=0.05,  # Less dropout for stability
                target_modules=[
                    "q_proj", "v_proj",  # Only attention, skip MLP for memory
                ],  # Minimal target modules for Mistral
            )
            logger.info("🎯 Ultra-efficient LoRA: r=8, attention-only, minimal parameters")
        else:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=16,  # Standard rank for other models
                lora_alpha=32,  # Standard scaling
                lora_dropout=0.1,
                target_modules=[
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],  # Full modules for other models
            )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        logger.info("✅ QLoRA setup complete - using ~4x less memory!")

    # Resize token embeddings if we added new tokens
    model.resize_token_embeddings(len(tokenizer))

    # CRITICAL FIX: Explicitly set pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"✅ Set model.config.pad_token_id to: {model.config.pad_token_id}")

    # For Mistral, also ensure the model knows about EOS token padding
    if 'mistral' in model_config.model_name.lower():
        model.config.pad_token_id = tokenizer.eos_token_id
        logger.info(f"🎯 Mistral: Set model.config.pad_token_id to EOS: {model.config.pad_token_id}")

    # Convert to HuggingFace datasets
    train_hf_dataset = Dataset.from_list([train_dataset[i] for i in range(len(train_dataset))])
    dev_hf_dataset = Dataset.from_list([dev_dataset[i] for i in range(len(dev_dataset))])

    # Memory management and batch size optimization
    effective_batch_size = training_config.batch_size

    # Ultra-conservative memory settings for Mistral 7B
    if use_qlora and ('mistral' in model_config.model_name.lower() or '7b' in model_config.model_name.lower()):
        effective_batch_size = 1  # Ultra conservative for Mistral
        logger.info("⚡ Mistral 7B QLoRA - ULTRA memory conservative: batch_size=1")
    elif use_qlora:
        effective_batch_size = 2  # Smaller than before for stability
        logger.info("🚀 QLoRA enabled - using batch_size=2")
    elif 'mistral' in model_config.model_name.lower() or '7b' in model_config.model_name.lower():
        effective_batch_size = 1
        logger.info("🔥 Large model detected - forcing batch_size=1 for memory")

    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.unk_token_id == -1:
        logger.warning("🚨 Padding token still problematic - forcing batch_size=1")
        effective_batch_size = 1

    # Training arguments - optimized for best model only
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        warmup_steps=training_config.warmup_steps,
        weight_decay=training_config.weight_decay,
        logging_dir=f"{training_config.output_dir}/logs",
        logging_steps=training_config.logging_steps,
        eval_steps=training_config.eval_steps,
        save_steps=training_config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",  # Enable wandb logging
        learning_rate=training_config.learning_rate,
        save_total_limit=1,  # Keep only 1 checkpoint (saves space!)
        dataloader_num_workers=0,  # Disable multiprocessing for memory
        bf16=torch.cuda.is_available(),  # Use bfloat16 for QLoRA compatibility
        remove_unused_columns=False,  # Prevent column removal issues
        dataloader_drop_last=False,  # Don't drop incomplete batches
        gradient_accumulation_steps=32 if (use_qlora and 'mistral' in model_config.model_name.lower()) else (4 if use_qlora else 8),  # High accumulation for Mistral
        gradient_checkpointing=False,  # Disable to avoid backward graph issues
        optim="paged_adamw_8bit" if (use_qlora and 'mistral' in model_config.model_name.lower()) else ("paged_adamw_32bit" if use_qlora else "adamw_torch"),  # 8-bit optimizer for Mistral
        dataloader_persistent_workers=False,  # Avoid worker issues
        include_inputs_for_metrics=False,  # Reduce memory overhead
        max_grad_norm=0.3 if 'mistral' in model_config.model_name.lower() else 1.0,  # Lower gradient clipping for stability
        adam_epsilon=1e-6 if 'mistral' in model_config.model_name.lower() else 1e-8,  # More stable for 4-bit
    )

    logger.info(f"Using batch sizes: train={effective_batch_size}, eval={effective_batch_size}")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=dev_hf_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Clear memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"🧹 Cleared CUDA cache. Available memory: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated():.2e} bytes")

    # Train model with error handling
    logger.info("Starting training...")
    try:
        trainer.train()
    except RuntimeError as e:
        if "backward through the graph" in str(e):
            logger.error("❌ Gradient checkpointing conflict detected. Restarting with safer settings...")
            # Force more conservative settings
            training_args.gradient_checkpointing = False
            training_args.dataloader_num_workers = 0
            training_args.per_device_train_batch_size = 1
            training_args.gradient_accumulation_steps = 16

            # Recreate trainer with safer settings
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_hf_dataset,
                eval_dataset=dev_hf_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            logger.info("🔄 Retrying with safer training settings...")
            trainer.train()
        else:
            raise e

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()

    # Save BEST model only (no huge checkpoints)
    saved_model_path = save_best_model_to_drive(trainer, tokenizer, training_config, train_dataset, eval_results, use_qlora)

    logger.info("Training completed!")
    logger.info(f"Final evaluation results: {eval_results}")
    logger.info(f"Best model saved to: {saved_model_path}")

    return trainer, eval_results


if __name__ == "__main__":
    # E.g. usage
    model_config = ModelConfig(model_name="bert-base-uncased")
    training_config = TrainingConfig()
    data_config = DataConfig()

    # Create output directory
    Path(training_config.output_dir).mkdir(exist_ok=True)

    # Train model
    trainer, results = train_model(model_config, training_config, data_config)