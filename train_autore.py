"""
AutoRE-inspired training script for document-level relation extraction.
Uses RHF (Relation-Head-Facts) paradigm with instruction-following approach.
"""
import json
import logging
import os
import torch
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Fix tokenizer parallelism warning and memory optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,  # Changed from SequenceClassification!
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,  # New: for text generation
)

from datasets import Dataset
import numpy as np

from config import ModelConfig, TrainingConfig, DataConfig
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import QLoRA dependencies
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


def load_relationship_descriptions(desc_file: str = "supplementary/data_cleaning/relationship_desc.json") -> Dict[str, str]:
    """Load relationship descriptions for instruction templates."""
    with open(desc_file, 'r') as f:
        return json.load(f)


def create_autore_instructions(examples: List[Dict], rel_descriptions: Dict[str, str]) -> List[Dict]:
    """Create AutoRE-style instruction-following dataset."""

    logger.info("Creating AutoRE instruction dataset...")
    instruction_data = []

    for example in examples:
        document = example['text']
        relation = example['relation']
        head_entity = example['head_entity']
        tail_entity = example['tail_entity']

        # Get relation description
        rel_desc = rel_descriptions.get(relation, f"The relation '{relation}' connects entities in a document.")

        # Stage 1: Relation Extraction Template
        stage1_instruction = f"""Given the following passage, identify all underlying relations that exist between entities:

Passage: {document}

Instructions: List any relations that you can identify in this passage. Focus on factual relationships between entities mentioned in the text.

Relations:"""

        stage1_response = f"{relation}"

        # Stage 2: Head Entity Identification Template
        stage2_instruction = f"""Given the relation "{relation}" and the following passage, identify all entities that could serve as the subject (head entity) for this relation:

Relation: {relation}
Description: {rel_desc}

Passage: {document}

Instructions: List entities that can be the subject of the "{relation}" relation.

Head entities:"""

        stage2_response = f"{head_entity}"

        # Stage 3: Fact Extraction Template
        stage3_instruction = f"""Given the relation "{relation}", the head entity "{head_entity}", and the following passage, extract the complete triplet facts:

Relation: {relation}
Head Entity: {head_entity}
Description: {rel_desc}

Passage: {document}

Instructions: List all complete triplet facts in the format (head, relation, tail) that involve the specified relation and head entity.

Facts:"""

        stage3_response = f"({head_entity}, {relation}, {tail_entity})"

        # Add all three stages to training data
        instruction_data.extend([
            {
                "instruction": stage1_instruction,
                "response": stage1_response,
                "stage": "relation_extraction",
                "relation": relation
            },
            {
                "instruction": stage2_instruction,
                "response": stage2_response,
                "stage": "head_identification",
                "relation": relation
            },
            {
                "instruction": stage3_instruction,
                "response": stage3_response,
                "stage": "fact_extraction",
                "relation": relation
            }
        ])

    logger.info(f"Created {len(instruction_data)} instruction examples from {len(examples)} original examples")
    return instruction_data


@dataclass
class AutoREDataset:
    """Dataset class for AutoRE instruction-following format."""

    def __init__(self, instruction_data: List[Dict], tokenizer, max_length: int = 1024):
        self.instruction_data = instruction_data
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Created AutoRE dataset with {len(instruction_data)} instruction examples")

    def __len__(self):
        return len(self.instruction_data)

    def __getitem__(self, idx):
        example = self.instruction_data[idx]

        # Format as instruction-following conversation
        instruction = example['instruction']
        response = example['response']

        # Create the full text for causal language modeling
        full_text = f"{instruction} {response}{self.tokenizer.eos_token}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()

        # Mask the instruction part - we only want to compute loss on the response
        instruction_tokens = self.tokenizer(
            instruction,
            truncation=True,
            max_length=self.max_length
        )['input_ids']

        # Set instruction tokens to -100 (ignored in loss computation)
        labels[:len(instruction_tokens)] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def load_processed_data(data_file: str) -> List[Dict]:
    """Load preprocessed data from JSON file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data['examples']


def compute_metrics(eval_pred):
    """Compute metrics for text generation evaluation - memory optimized."""
    predictions, labels = eval_pred

    # MEMORY FIX: Don't store large tensors, compute metrics incrementally
    if predictions is None or labels is None:
        return {'eval_loss': 0.0, 'perplexity': 1.0}

    # Convert to numpy to save GPU memory
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    # For AutoRE, we mainly care about loss during training
    # The real evaluation happens in evaluate_autore.py
    try:
        # Simple perplexity approximation without storing large tensors
        mask = labels != -100
        if mask.sum() == 0:
            return {'eval_loss': 0.0, 'perplexity': 1.0}

        # Sample-based evaluation to avoid memory issues
        sample_size = min(1000, mask.sum())  # Limit to 1000 tokens for memory
        valid_indices = np.where(mask.flatten())[0][:sample_size]

        if len(valid_indices) == 0:
            return {'eval_loss': 0.0, 'perplexity': 1.0}

        # Compute approximate perplexity on sample
        sample_predictions = predictions.reshape(-1, predictions.shape[-1])[valid_indices]
        sample_labels = labels.flatten()[valid_indices]

        # Compute cross-entropy on CPU to save GPU memory
        loss = torch.nn.functional.cross_entropy(
            torch.tensor(sample_predictions, dtype=torch.float32),
            torch.tensor(sample_labels, dtype=torch.long),
            reduction='mean'
        )

        perplexity = torch.exp(loss).item()

        # Clear tensors immediately
        del sample_predictions, sample_labels, predictions, labels
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return {
            'perplexity': perplexity,
            'eval_loss': loss.item()
        }

    except Exception as e:
        logger.warning(f"Metrics computation failed: {e}, returning defaults")
        return {'eval_loss': 0.0, 'perplexity': 1.0}


def save_autore_model(trainer, tokenizer, training_config, eval_results, use_qlora=False):
    """Save AutoRE model with proper metadata."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Try to mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        drive_available = True
        drive_path = f"/content/drive/MyDrive/AutoRE_Models/autore_model_{timestamp}"
    except:
        logger.warning("Google Drive not available, saving locally only")
        drive_available = False
        drive_path = None

    # Save the model
    logger.info(f"Saving AutoRE model to {training_config.output_dir}")

    if use_qlora:
        # For QLoRA, save only the adapter weights
        logger.info("💾 Saving AutoRE QLoRA adapter...")
        trainer.model.save_pretrained(training_config.output_dir)

        # Save proper adapter config for AutoRE
        adapter_config = {
            "base_model_name_or_path": trainer.model.peft_config['default'].base_model_name_or_path,
            "model_type": "autore_qlora_adapter",
            "training_paradigm": "RHF",
            "task_type": "CAUSAL_LM"
        }

        with open(Path(training_config.output_dir) / "adapter_config.json", 'w') as f:
            json.dump(adapter_config, f, indent=2)

        logger.info("✅ AutoRE QLoRA adapter saved!")
    else:
        # Standard model saving
        trainer.save_model(training_config.output_dir)

    tokenizer.save_pretrained(training_config.output_dir)

    # Create metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'AutoRE_RHF',
        'training_paradigm': 'Relation-Head-Facts',
        'base_model': training_config.output_dir,
        'training_config': {
            'batch_size': training_config.batch_size,
            'learning_rate': training_config.learning_rate,
            'num_epochs': training_config.num_epochs,
        },
        'eval_results': eval_results,
        'supported_relations': 96
    }

    # Save metadata
    with open(Path(training_config.output_dir) / "autore_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Copy to Google Drive if available
    if drive_available:
        try:
            Path(drive_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Copying AutoRE model to Google Drive: {drive_path}")

            # Copy essential files
            essential_files = [
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "autore_metadata.json", "special_tokens_map.json"
            ]

            if use_qlora:
                essential_files.extend(["adapter_config.json", "adapter_model.safetensors"])
            else:
                essential_files.extend(["pytorch_model.bin", "model.safetensors"])

            for file_name in essential_files:
                src_file = Path(training_config.output_dir) / file_name
                if src_file.exists():
                    shutil.copy(src_file, Path(drive_path) / file_name)
                    logger.info(f"✅ Copied {file_name} to Drive")

            drive_success = True
        except Exception as e:
            logger.error(f"Failed to copy to Google Drive: {e}")
            drive_success = False
    else:
        drive_success = False

    # Create summary
    summary = f"""
🎯 AutoRE Model Training Complete!
=================================

📅 Timestamp: {timestamp}
🧠 Model Type: AutoRE RHF (Relation-Head-Facts)
📁 Local Location: {training_config.output_dir}
{'💾 Google Drive: ' + drive_path if drive_success else '❌ Google Drive: Failed'}

📊 Results:
- Perplexity: {eval_results.get('eval_perplexity', 'N/A'):.4f}
- Loss: {eval_results.get('eval_loss', 'N/A'):.4f}

🏷️ Model Info:
- Supported relations: 96
- Training paradigm: RHF (3-stage instruction following)
- Base architecture: Causal Language Model

💡 To use this model:
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('{drive_path if drive_success else training_config.output_dir}')
model = AutoModelForCausalLM.from_pretrained('{drive_path if drive_success else training_config.output_dir}')

✨ AutoRE RHF model ready for relation extraction!
"""

    with open("./autore_training_summary.txt", 'w') as f:
        f.write(summary)

    print(summary)
    logger.info(f"AutoRE training summary saved to ./autore_training_summary.txt")

    return drive_path if drive_success else training_config.output_dir


def train_autore_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig
):
    """Main AutoRE training function using RHF paradigm."""

    logger.info("🚀 Starting AutoRE RHF Training Pipeline")

    # Load relationship descriptions
    rel_descriptions = load_relationship_descriptions()
    logger.info(f"Loaded {len(rel_descriptions)} relationship descriptions")

    # Load data
    logger.info("Loading training data...")
    train_examples = load_processed_data(Path(data_config.data_dir) / "train_processed.json")
    dev_examples = load_processed_data(Path(data_config.data_dir) / "dev_processed.json")

    # Limit examples if specified
    if data_config.max_examples:
        train_examples = train_examples[:data_config.max_examples]
        dev_examples = dev_examples[:min(data_config.max_examples // 4, len(dev_examples))]

    # MEMORY FIX: Limit dev examples further to prevent eval OOM
    # Since each example creates 3 instruction examples, we need to be conservative
    max_dev_for_memory = min(500, len(dev_examples))  # Max 500 dev examples = 1500 instruction examples
    dev_examples = dev_examples[:max_dev_for_memory]

    logger.info(f"Training examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(dev_examples)} (limited for memory)")

    # Create instruction-following dataset
    train_instructions = create_autore_instructions(train_examples, rel_descriptions)
    dev_instructions = create_autore_instructions(dev_examples, rel_descriptions)

    logger.info(f"Created {len(train_instructions)} training instructions")
    logger.info(f"Created {len(dev_instructions)} validation instructions (memory-limited)")

    # Load tokenizer and model
    logger.info(f"Loading model: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    # Fix padding token for causal LM
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Create datasets
    train_dataset = AutoREDataset(train_instructions, tokenizer, model_config.max_length)
    dev_dataset = AutoREDataset(dev_instructions, tokenizer, model_config.max_length)

    # Load model with QLoRA optimizations
    use_qlora = QLORA_AVAILABLE and ('mistral' in model_config.model_name.lower() or '7b' in model_config.model_name.lower())

    if use_qlora:
        logger.info("🚀 Using QLoRA for AutoRE training!")

        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(  # Changed to CausalLM!
            model_config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration for causal LM
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Changed from SEQ_CLS!
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        logger.info("✅ QLoRA setup complete for AutoRE!")

    else:
        logger.info("🔧 Loading model with standard optimizations...")
        model = AutoModelForCausalLM.from_pretrained(  # Changed to CausalLM!
            model_config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # Resize token embeddings if we added new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Convert to HuggingFace datasets
    train_hf_dataset = Dataset.from_list([train_dataset[i] for i in range(len(train_dataset))])
    dev_hf_dataset = Dataset.from_list([dev_dataset[i] for i in range(len(dev_dataset))])

    # MEMORY FIX: Clear intermediate datasets
    del train_dataset, dev_dataset, train_instructions, dev_instructions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("🧹 Cleared intermediate datasets from memory")

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not masked LM, it's causal LM
    )

    # Training arguments optimized for AutoRE with memory fixes
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=2 if use_qlora else 4,
        per_device_eval_batch_size=1,  # REDUCED: Smaller eval batch to save memory
        warmup_steps=training_config.warmup_steps,
        weight_decay=training_config.weight_decay,
        logging_dir=f"{training_config.output_dir}/logs",
        logging_steps=50,
        eval_steps=500,
        save_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use loss for LM
        greater_is_better=False,  # Lower loss is better
        report_to="wandb",
        learning_rate=1e-5,  # Conservative LR for instruction following
        save_total_limit=2,
        dataloader_num_workers=0,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        optim="paged_adamw_32bit" if use_qlora else "adamw_torch",
        # MEMORY FIXES FOR EVALUATION:
        eval_accumulation_steps=1,  # Process eval in smaller chunks
        prediction_loss_only=True,  # Don't store predictions, only compute loss
        dataloader_pin_memory=False,  # Reduce memory pressure
    )

    logger.info(f"Using batch size: {training_args.per_device_train_batch_size}")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=dev_hf_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 Cleared CUDA cache")

    # Train model
    logger.info("🎯 Starting AutoRE RHF training...")
    trainer.train()

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()

    # Save model
    saved_model_path = save_autore_model(trainer, tokenizer, training_config, eval_results, use_qlora)

    logger.info("AutoRE training completed!")
    logger.info(f"Final evaluation results: {eval_results}")
    logger.info(f"AutoRE model saved to: {saved_model_path}")

    return trainer, eval_results


if __name__ == "__main__":
    # Example usage
    model_config = ModelConfig(model_name="mistralai/Mistral-7B-v0.3")
    training_config = TrainingConfig()
    data_config = DataConfig()

    # Create output directory
    Path(training_config.output_dir).mkdir(exist_ok=True)

    # Train AutoRE model
    trainer, results = train_autore_model(model_config, training_config, data_config)