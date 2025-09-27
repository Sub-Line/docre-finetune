"""
Configuration settings for the RE-DocRED finetuning project.
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name: str = "bert-base-uncased"  # Default model, will be user-configurable
    max_length: int = 512
    num_labels: int = 97  # Number of relation types in RE-DocRED
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./outputs"

@dataclass
class DataConfig:
    """Configuration for data settings."""
    data_dir: str = "./data"
    train_file: str = "train_annotated.json"
    dev_file: str = "dev.json"
    test_file: str = "test.json"
    max_examples: Optional[int] = None  # For testing with smaller datasets

@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace Hub settings."""
    repo_name: Optional[str] = None
    organization: Optional[str] = None
    private: bool = False
    token: Optional[str] = None