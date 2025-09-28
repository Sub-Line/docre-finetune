"""
Evaluation script for trained relation extraction models.
"""
import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime

from train import RelationExtractionDataset, load_processed_data
from config import ModelConfig, DataConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trained_model(model_path: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, Dict]:
    """Load trained model, tokenizer, and label mapping."""
    logger.info(f"Loading model from {model_path}")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load label mapping
    label_mapping_path = Path(model_path) / "label_mapping.json"
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    return model, tokenizer, label_mapping


def predict_batch(model, tokenizer, examples: List[Dict], label_mapping: Dict, max_length: int = 512) -> List[str]:
    """Make predictions on a batch of examples."""
    model.eval()

    predictions = []
    id2label = label_mapping['id2label']

    with torch.no_grad():
        for example in examples:
            # Format input text
            text = example['text']
            head_entity = example['head_entity']
            tail_entity = example['tail_entity']
            input_text = f"{text} [SEP] {head_entity} [SEP] {tail_entity}"

            # Tokenize
            encoding = tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            # Predict
            outputs = model(**encoding)
            predicted_label_id = torch.argmax(outputs.logits, dim=-1).item()
            predicted_label = id2label[str(predicted_label_id)]

            predictions.append(predicted_label)

    return predictions


def evaluate_model(model_path: str, test_data_path: str, data_config: DataConfig) -> Dict[str, Any]:
    """Evaluate trained model on test data."""
    logger.info("Starting model evaluation...")

    # Load model
    model, tokenizer, label_mapping = load_trained_model(model_path)

    # Load test data
    test_examples = load_processed_data(test_data_path)
    logger.info(f"Loaded {len(test_examples)} test examples")

    # Limit examples if specified
    if data_config.max_examples:
        test_examples = test_examples[:data_config.max_examples]

    # Get true labels
    true_labels = [example['relation'] for example in test_examples]

    # Make predictions
    logger.info("Making predictions...")
    predicted_labels = predict_batch(model, tokenizer, test_examples, label_mapping)

    # Calculate metrics
    logger.info("Calculating metrics...")

    # Get unique labels for classification report
    unique_labels = sorted(list(set(true_labels + predicted_labels)))

    # Classification report
    class_report = classification_report(
        true_labels,
        predicted_labels,
        labels=unique_labels,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

    # Overall accuracy
    accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / len(true_labels)

    # Collect results
    results = {
        'accuracy': accuracy,
        'macro_avg_f1': class_report['macro avg']['f1-score'],
        'weighted_avg_f1': class_report['weighted avg']['f1-score'],
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'label_names': unique_labels,
        'num_test_examples': len(test_examples),
        'num_predictions': len(predicted_labels)
    }

    # Log summary
    logger.info(f"Evaluation Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Macro F1: {class_report['macro avg']['f1-score']:.4f}")
    logger.info(f"  Weighted F1: {class_report['weighted avg']['f1-score']:.4f}")

    return results


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Evaluation results saved to {output_path}")


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed evaluation results."""
    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS")
    print("="*80)

    print(f"\nOverall Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_avg_f1']:.4f}")
    print(f"  Weighted F1: {results['weighted_avg_f1']:.4f}")

    print(f"\nPer-Class Performance:")
    class_report = results['classification_report']
    for label in results['label_names']:
        if label in class_report:
            metrics = class_report[label]
            print(f"  {label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")
            print(f"    Support: {metrics['support']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage
    model_path = "./outputs"
    test_data_path = "./data/test_processed.json"

    data_config = DataConfig()

    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model path does not exist: {model_path}")
        exit(1)

    # Check if test data exists
    if not Path(test_data_path).exists():
        logger.error(f"Test data does not exist: {test_data_path}")
        exit(1)

    # Evaluate model
    results = evaluate_model(model_path, test_data_path, data_config)

    # Print detailed results
    print_detailed_results(results)

    # Save results to multiple locations for Colab safety
    output_path = Path(model_path) / "evaluation_results.json"
    save_evaluation_results(results, str(output_path))

    # Also save to backups directory if it exists
    backup_pattern = "./backups/model_*"
    import glob
    backup_dirs = glob.glob(backup_pattern)
    if backup_dirs:
        latest_backup = max(backup_dirs)  # Get most recent backup
        backup_eval_path = Path(latest_backup) / "evaluation_results.json"
        save_evaluation_results(results, str(backup_eval_path))
        logger.info(f"Evaluation results also saved to backup: {backup_eval_path}")

    # Save summary for easy access
    summary_text = f"""
📊 Model Evaluation Summary
==========================

📁 Model Path: {model_path}
📅 Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 Performance Metrics:
- Accuracy: {results['accuracy']:.4f}
- Macro F1: {results['macro_avg_f1']:.4f}
- Weighted F1: {results['weighted_avg_f1']:.4f}

📈 Dataset Info:
- Test Examples: {results['num_test_examples']}
- Relation Types: {len(results['label_names'])}

💾 Results saved to:
- {output_path}
{"- " + str(backup_eval_path) if backup_dirs else ""}
"""

    with open("./evaluation_summary.txt", 'w') as f:
        f.write(summary_text)

    print(summary_text)