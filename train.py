"""
Training script for finetuning LLMs on RE-DocRED for relation extraction.
Uses HuggingFace Transformers with Trainer API for model-agnostic training.
"""
import json
import logging
import os
import torch
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

from config import ModelConfig, TrainingConfig, DataConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RelationExtractionDataset:
    """Custom dataset class for relation extraction."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create label mapping
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

    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    train_dataset = RelationExtractionDataset(train_examples, tokenizer, model_config.max_length)
    dev_dataset = RelationExtractionDataset(dev_examples, tokenizer, model_config.max_length)

    # Ensure both datasets use the same label mapping
    dev_dataset.label2id = train_dataset.label2id
    dev_dataset.id2label = train_dataset.id2label

    # Load model with correct number of labels
    num_labels = len(train_dataset.label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name,
        num_labels=num_labels,
        id2label=train_dataset.id2label,
        label2id=train_dataset.label2id
    )

    # Resize token embeddings if we added new tokens
    model.resize_token_embeddings(len(tokenizer))

    # Convert to HuggingFace datasets
    train_hf_dataset = Dataset.from_list([train_dataset[i] for i in range(len(train_dataset))])
    dev_hf_dataset = Dataset.from_list([dev_dataset[i] for i in range(len(dev_dataset))])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
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
        report_to=None,  # Disable wandb by default
        learning_rate=training_config.learning_rate,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=dev_hf_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Save model and tokenizer
    logger.info(f"Saving model to {training_config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_config.output_dir)

    # Save label mapping
    with open(Path(training_config.output_dir) / "label_mapping.json", 'w') as f:
        json.dump({
            'label2id': train_dataset.label2id,
            'id2label': train_dataset.id2label
        }, f, indent=2)

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()

    logger.info("Training completed!")
    logger.info(f"Final evaluation results: {eval_results}")

    return trainer, eval_results


if __name__ == "__main__":
    # Example usage
    model_config = ModelConfig(model_name="bert-base-uncased")
    training_config = TrainingConfig()
    data_config = DataConfig()

    # Create output directory
    Path(training_config.output_dir).mkdir(exist_ok=True)

    # Train model
    trainer, results = train_model(model_config, training_config, data_config)