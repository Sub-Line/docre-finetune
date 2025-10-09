"""
Configuration settings for finetuning (training, data and HF).
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name: str = "bert-base-uncased"  # default model
    max_length: int = 512
    num_labels: int = 97  # num of relation types in RE-DocRED
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3

    def __post_init__(self):
        # Auto-adjust batch size for large models to prevent OOM
        if hasattr(self, '_model_name'):
            if 'mistral' in self._model_name.lower() or '7b' in self._model_name.lower():
                self.batch_size = min(self.batch_size, 4)  # Reduce for large models
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
    max_examples: Optional[int] = None  # can use this for testing with smaller datasets

@dataclass
class HuggingFaceConfig:
    """Configuration for HF Hub settings."""
    repo_name: Optional[str] = None
    organization: Optional[str] = None
    private: bool = False
    token: Optional[str] = None