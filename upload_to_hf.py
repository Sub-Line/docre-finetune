"""
Upload trained model to HuggingFace Hub with user approval.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import HfHubHTTPError

from config import HuggingFaceConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_user_approval_and_config(eval_results: Dict[str, Any]) -> Optional[HuggingFaceConfig]:
    """Get user approval and configuration for uploading to HF Hub."""
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    print(f"Accuracy: {eval_results.get('accuracy', 0):.4f}")
    print(f"Macro F1: {eval_results.get('macro_avg_f1', 0):.4f}")
    print(f"Weighted F1: {eval_results.get('weighted_avg_f1', 0):.4f}")
    print(f"Number of test examples: {eval_results.get('num_test_examples', 0)}")
    print("="*80)

    # Ask for user approval
    while True:
        upload_choice = input("\nDo you want to upload this model to HuggingFace Hub? (y/n): ").strip().lower()
        if upload_choice in ['y', 'yes']:
            break
        elif upload_choice in ['n', 'no']:
            print("Upload cancelled by user.")
            return None
        else:
            print("Please enter 'y' or 'n'.")

    # Get HF configuration
    print("\nHuggingFace Hub Configuration:")

    # Repository name
    repo_name = input("Enter repository name (e.g., 'my-relation-extraction-model'): ").strip()
    if not repo_name:
        print("Repository name is required.")
        return None

    # Organization (optional)
    organization = input("Enter organization name (optional, press Enter to skip): ").strip()
    if not organization:
        organization = None

    # Private repository
    while True:
        private_choice = input("Make repository private? (y/n): ").strip().lower()
        if private_choice in ['y', 'yes']:
            private = True
            break
        elif private_choice in ['n', 'no']:
            private = False
            break
        else:
            print("Please enter 'y' or 'n'.")

    # HF Token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        hf_token = input("Enter your HuggingFace token (or set HF_TOKEN environment variable): ").strip()
        if not hf_token:
            print("HuggingFace token is required for upload.")
            return None

    return HuggingFaceConfig(
        repo_name=repo_name,
        organization=organization,
        private=private,
        token=hf_token
    )


def create_model_card(
    model_path: str,
    eval_results: Dict[str, Any],
    hf_config: HuggingFaceConfig,
    model_name: str
) -> str:
    """Create a model card for the uploaded model."""

    # Load label mapping for relation types
    label_mapping_path = Path(model_path) / "label_mapping.json"
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    num_relations = len(label_mapping['label2id'])
    relation_types = sorted(label_mapping['label2id'].keys())

    model_card_content = f"""---
language: en
license: mit
tags:
- relation-extraction
- knowledge-graph
- docred
- re-docred
- information-extraction
datasets:
- re-docred
metrics:
- accuracy
- f1
model-index:
- name: {hf_config.repo_name}
  results:
  - task:
      type: relation-extraction
      name: Relation Extraction
    dataset:
      type: re-docred
      name: RE-DocRED
    metrics:
    - type: accuracy
      value: {eval_results.get('accuracy', 0):.4f}
    - type: f1_macro
      value: {eval_results.get('macro_avg_f1', 0):.4f}
    - type: f1_weighted
      value: {eval_results.get('weighted_avg_f1', 0):.4f}
---

# {hf_config.repo_name}

This model is fine-tuned on the RE-DocRED dataset for document-level relation extraction. It can extract relationships between entities in documents and is designed for building knowledge graphs from text.

## Model Details

- **Base Model**: {model_name}
- **Task**: Relation Extraction
- **Dataset**: RE-DocRED
- **Number of Relation Types**: {num_relations}

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | {eval_results.get('accuracy', 0):.4f} |
| Macro F1 | {eval_results.get('macro_avg_f1', 0):.4f} |
| Weighted F1 | {eval_results.get('weighted_avg_f1', 0):.4f} |

## Supported Relations

The model can classify the following {num_relations} relation types:

{chr(10).join([f"- {relation}" for relation in relation_types[:20]])}
{"..." if len(relation_types) > 20 else ""}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "{hf_config.organization + '/' if hf_config.organization else ''}{hf_config.repo_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example usage
text = "Your document text here."
head_entity = "Entity 1"
tail_entity = "Entity 2"

# Format input
input_text = f"{{text}} [SEP] {{head_entity}} [SEP] {{tail_entity}}"

# Tokenize and predict
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=-1)

# Get relation label
relation = model.config.id2label[predicted_class.item()]
print(f"Predicted relation: {{relation}}")
```

## Training Data

This model was trained on the RE-DocRED dataset, which is a revised version of DocRED that addresses:
- Incompleteness by adding missing relation triples
- Logical inconsistencies in annotations
- Coreferential errors

## Citation

If you use this model, please cite:

```bibtex
@inproceedings{{tan-etal-2022-revisiting,
    title = "Revisiting DocRED -- Addressing the False Negative Problem in Relation Extraction",
    author = "Tan, Qingyu and He, Ruidan and Bing, Lidong and Ng, Hwee Tou",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022"
}}
```

## Disclaimer

This model is intended for research and educational purposes. Please ensure appropriate use and consider potential biases in the training data.
"""

    return model_card_content


def upload_model_to_hf(
    model_path: str,
    eval_results: Dict[str, Any],
    hf_config: HuggingFaceConfig,
    base_model_name: str
) -> bool:
    """Upload model to HuggingFace Hub."""
    try:
        # Initialize HF API and validate token
        api = HfApi(token=hf_config.token)

        # Validate token works
        try:
            user_info = api.whoami()
            logger.info(f"✅ Token validated for user: {user_info['name']}")
        except Exception as e:
            logger.error(f"❌ Token validation failed: {e}")
            logger.error("Please check your HuggingFace token permissions")
            return False

        # Create repository name
        repo_id = f"{hf_config.organization}/{hf_config.repo_name}" if hf_config.organization else hf_config.repo_name

        logger.info(f"Creating repository: {repo_id}")

        # Create repository with better error handling
        try:
            logger.info(f"Attempting to create repository: {repo_id}")
            logger.info(f"Repository settings - Private: {hf_config.private}, Token available: {bool(hf_config.token)}")

            create_repo(
                repo_id=repo_id,
                token=hf_config.token,
                private=hf_config.private,
                exist_ok=True,
                repo_type="model"  # Explicitly specify model type
            )
            logger.info(f"✅ Repository created/verified: {repo_id}")

            # Verify repository exists by checking with API
            try:
                api.repo_info(repo_id=repo_id, token=hf_config.token)
                logger.info(f"✅ Repository confirmed accessible: {repo_id}")
            except Exception as e:
                logger.warning(f"⚠️ Repository verification failed: {e}")

        except Exception as e:
            logger.error(f"❌ Repository creation failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            if "401" in str(e) or "authentication" in str(e).lower():
                logger.error("Authentication error - check your HF token permissions")
            elif "403" in str(e) or "forbidden" in str(e).lower():
                logger.error("Permission error - token may not have write access")
            elif "already exists" in str(e).lower():
                logger.info(f"Repository already exists: {repo_id}")
            else:
                raise e

        # Create and save model card
        logger.info("Creating model card...")
        model_card_content = create_model_card(model_path, eval_results, hf_config, base_model_name)
        readme_path = Path(model_path) / "README.md"
        with open(readme_path, 'w') as f:
            f.write(model_card_content)

        # Upload model files with retry mechanism
        logger.info(f"Uploading model files from {model_path}...")
        try:
            upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                token=hf_config.token,
                ignore_patterns=["*.log", "__pycache__", "*.pyc", "checkpoint-*"],
                commit_message="Upload RE-DocRED fine-tuned model",
                commit_description="Fine-tuned model for relation extraction on RE-DocRED dataset"
            )
            logger.info(f"✅ Model files uploaded successfully to {repo_id}")
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            if "Repository not found" in str(e):
                logger.error("Repository not found - trying to create it again...")
                # Retry repository creation
                create_repo(
                    repo_id=repo_id,
                    token=hf_config.token,
                    private=hf_config.private,
                    exist_ok=True,
                    repo_type="model"
                )
                logger.info("Repository created, retrying upload...")
                upload_folder(
                    folder_path=model_path,
                    repo_id=repo_id,
                    token=hf_config.token,
                    ignore_patterns=["*.log", "__pycache__", "*.pyc", "checkpoint-*"],
                    commit_message="Upload RE-DocRED fine-tuned model"
                )
                logger.info(f"✅ Model files uploaded successfully on retry to {repo_id}")
            else:
                raise e

        logger.info(f"Model successfully uploaded to: https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        return False


def auto_upload_workflow(model_path: str, base_model_name: str):
    """Automatic upload workflow using default settings (pre-approved)."""
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False

    # Load evaluation results
    eval_results_path = Path(model_path) / "evaluation_results.json"
    if eval_results_path.exists():
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
    else:
        logger.warning("No evaluation results found. Using default values.")
        eval_results = {
            'accuracy': 0.0,
            'macro_avg_f1': 0.0,
            'weighted_avg_f1': 0.0,
            'num_test_examples': 0
        }

    # Display results but use default config
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS (AUTO-UPLOAD)")
    print("="*80)
    print(f"Accuracy: {eval_results.get('accuracy', 0):.4f}")
    print(f"Macro F1: {eval_results.get('macro_avg_f1', 0):.4f}")
    print(f"Weighted F1: {eval_results.get('weighted_avg_f1', 0):.4f}")
    print(f"Number of test examples: {eval_results.get('num_test_examples', 0)}")
    print("="*80)

    # Use default HF configuration (pre-approved)
    # Clean model name for repo naming
    clean_model_name = base_model_name.replace("/", "-").replace("_", "-").lower()
    default_repo_name = f"re-docred-{clean_model_name}-finetuned"

    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.error("HF_TOKEN environment variable required for auto-upload")
        print("❌ Auto-upload failed: HF_TOKEN environment variable not set")
        print("💡 Set HF_TOKEN or use 'ask_later' mode for interactive upload")
        return False

    hf_config = HuggingFaceConfig(
        repo_name=default_repo_name,
        organization=None,  # Upload to personal account
        private=False,      # Public by default
        token=hf_token
    )

    print(f"🚀 Auto-uploading to: {hf_config.repo_name}")
    print(f"📂 Repository will be: Public")
    print(f"🏢 Organization: Personal account")

    # Upload model
    success = upload_model_to_hf(model_path, eval_results, hf_config, base_model_name)

    if success:
        print(f"\n✅ Model automatically uploaded!")
        print(f"🔗 Model URL: https://huggingface.co/{hf_config.repo_name}")
        print(f"📚 You can now use this model with: transformers.AutoModel.from_pretrained('{hf_config.repo_name}')")
    else:
        print("\n❌ Auto-upload failed. Please check the logs for details.")

    return success


def main_upload_workflow(model_path: str, base_model_name: str):
    """Main workflow for uploading model to HF Hub."""
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False

    # Load evaluation results
    eval_results_path = Path(model_path) / "evaluation_results.json"
    if eval_results_path.exists():
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
    else:
        logger.warning("No evaluation results found. Using default values.")
        eval_results = {
            'accuracy': 0.0,
            'macro_avg_f1': 0.0,
            'weighted_avg_f1': 0.0,
            'num_test_examples': 0
        }

    # Get user approval and configuration
    hf_config = get_user_approval_and_config(eval_results)
    if not hf_config:
        return False

    # Upload model
    success = upload_model_to_hf(model_path, eval_results, hf_config, base_model_name)

    if success:
        print(f"\n✅ Model successfully uploaded!")
        repo_id = f"{hf_config.organization}/{hf_config.repo_name}" if hf_config.organization else hf_config.repo_name
        print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
        print(f"📚 You can now use this model with: transformers.AutoModel.from_pretrained('{repo_id}')")
    else:
        print("\n❌ Upload failed. Please check the logs for details.")

    return success


if __name__ == "__main__":
    # Example usage
    model_path = "./outputs"
    base_model_name = "bert-base-uncased"  # This should be passed from the training script

    main_upload_workflow(model_path, base_model_name)