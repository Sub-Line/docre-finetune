"""
Evaluation script for AutoRE models using text generation approach.
Evaluates the RHF (Relation-Head-Facts) paradigm performance.
"""
import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_autore_model(model_path: str) -> Tuple[Any, Any]:
    """Load AutoRE model (supports both QLoRA and full models)."""

    logger.info(f"Loading AutoRE model from: {model_path}")

    try:
        # Check if it's a PEFT model
        try:
            peft_config = PeftConfig.from_pretrained(model_path)
            logger.info("📦 Detected PEFT/LoRA model")

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            # Load adapter
            model = PeftModel.from_pretrained(base_model, model_path)
            logger.info("✅ PEFT model loaded successfully")

        except:
            # Regular model
            logger.info("📦 Loading as regular CausalLM model")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("✅ AutoRE model loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 100) -> str:
    """Generate response from instruction using the AutoRE model."""

    # Tokenize instruction
    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate response
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode response (remove the input part)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(instruction):].strip()

    return response


def evaluate_relation_extraction(model, tokenizer, test_examples: List[Dict], rel_descriptions: Dict[str, str]) -> Dict[str, float]:
    """Evaluate Stage 1: Relation Extraction."""

    logger.info("🎯 Evaluating Stage 1: Relation Extraction")

    correct = 0
    total = 0
    relation_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for example in test_examples:
        document = example['text']
        true_relation = example['relation']

        # Create instruction
        instruction = f"""Given the following passage, identify all underlying relations that exist between entities:

Passage: {document}

Instructions: List any relations that you can identify in this passage. Focus on factual relationships between entities mentioned in the text.

Relations:"""

        # Generate response
        response = generate_response(model, tokenizer, instruction, max_new_tokens=50)

        # Check if true relation is mentioned in response
        if true_relation.lower() in response.lower():
            correct += 1
            relation_stats[true_relation]["correct"] += 1

        relation_stats[true_relation]["total"] += 1
        total += 1

        if total % 100 == 0:
            logger.info(f"Processed {total} examples for relation extraction")

    accuracy = correct / total if total > 0 else 0
    logger.info(f"✅ Relation Extraction Accuracy: {accuracy:.4f}")

    return {
        "relation_accuracy": accuracy,
        "relation_stats": dict(relation_stats)
    }


def evaluate_head_identification(model, tokenizer, test_examples: List[Dict], rel_descriptions: Dict[str, str]) -> Dict[str, float]:
    """Evaluate Stage 2: Head Entity Identification."""

    logger.info("🎯 Evaluating Stage 2: Head Entity Identification")

    correct = 0
    total = 0

    for example in test_examples:
        document = example['text']
        relation = example['relation']
        true_head = example['head_entity']

        # Get relation description
        rel_desc = rel_descriptions.get(relation, f"The relation '{relation}' connects entities in a document.")

        # Create instruction
        instruction = f"""Given the relation "{relation}" and the following passage, identify all entities that could serve as the subject (head entity) for this relation:

Relation: {relation}
Description: {rel_desc}

Passage: {document}

Instructions: List entities that can be the subject of the "{relation}" relation.

Head entities:"""

        # Generate response
        response = generate_response(model, tokenizer, instruction, max_new_tokens=100)

        # Check if true head entity is mentioned
        if true_head.lower() in response.lower():
            correct += 1

        total += 1

        if total % 100 == 0:
            logger.info(f"Processed {total} examples for head identification")

    accuracy = correct / total if total > 0 else 0
    logger.info(f"✅ Head Identification Accuracy: {accuracy:.4f}")

    return {"head_accuracy": accuracy}


def evaluate_fact_extraction(model, tokenizer, test_examples: List[Dict], rel_descriptions: Dict[str, str]) -> Dict[str, float]:
    """Evaluate Stage 3: Complete Fact Extraction."""

    logger.info("🎯 Evaluating Stage 3: Fact Extraction")

    correct_exact = 0
    correct_partial = 0
    total = 0

    for example in test_examples:
        document = example['text']
        relation = example['relation']
        head_entity = example['head_entity']
        true_tail = example['tail_entity']

        # Get relation description
        rel_desc = rel_descriptions.get(relation, f"The relation '{relation}' connects entities in a document.")

        # Create instruction
        instruction = f"""Given the relation "{relation}", the head entity "{head_entity}", and the following passage, extract the complete triplet facts:

Relation: {relation}
Head Entity: {head_entity}
Description: {rel_desc}

Passage: {document}

Instructions: List all complete triplet facts in the format (head, relation, tail) that involve the specified relation and head entity.

Facts:"""

        # Generate response
        response = generate_response(model, tokenizer, instruction, max_new_tokens=150)

        # Check for exact triplet match
        expected_triplet = f"({head_entity}, {relation}, {true_tail})"
        if expected_triplet.lower() in response.lower():
            correct_exact += 1
            correct_partial += 1
        # Check for partial match (tail entity present)
        elif true_tail.lower() in response.lower():
            correct_partial += 1

        total += 1

        if total % 100 == 0:
            logger.info(f"Processed {total} examples for fact extraction")

    exact_accuracy = correct_exact / total if total > 0 else 0
    partial_accuracy = correct_partial / total if total > 0 else 0

    logger.info(f"✅ Exact Fact Extraction Accuracy: {exact_accuracy:.4f}")
    logger.info(f"✅ Partial Fact Extraction Accuracy: {partial_accuracy:.4f}")

    return {
        "fact_exact_accuracy": exact_accuracy,
        "fact_partial_accuracy": partial_accuracy
    }


def load_processed_data(data_file: str) -> List[Dict]:
    """Load preprocessed data from JSON file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data['examples']


def load_relationship_descriptions(desc_file: str = "supplementary/data_cleaning/relationship_desc.json") -> Dict[str, str]:
    """Load relationship descriptions."""
    with open(desc_file, 'r') as f:
        return json.load(f)


def evaluate_autore_model(model_path: str, test_data_path: str, data_config: Any) -> Dict[str, Any]:
    """Main evaluation function for AutoRE model."""

    logger.info(f"🔍 Starting AutoRE model evaluation")
    logger.info(f"Model: {model_path}")
    logger.info(f"Test data: {test_data_path}")

    # Load model
    model, tokenizer = load_autore_model(model_path)

    # Load test data
    test_examples = load_processed_data(test_data_path)
    logger.info(f"Loaded {len(test_examples)} test examples")

    # Limit test examples if specified
    if hasattr(data_config, 'max_test_examples') and data_config.max_test_examples:
        test_examples = test_examples[:data_config.max_test_examples]
        logger.info(f"Limited to {len(test_examples)} test examples")

    # Load relationship descriptions
    rel_descriptions = load_relationship_descriptions()

    # Run all three evaluation stages
    results = {}

    # Stage 1: Relation Extraction
    stage1_results = evaluate_relation_extraction(model, tokenizer, test_examples, rel_descriptions)
    results.update(stage1_results)

    # Stage 2: Head Entity Identification
    stage2_results = evaluate_head_identification(model, tokenizer, test_examples, rel_descriptions)
    results.update(stage2_results)

    # Stage 3: Fact Extraction
    stage3_results = evaluate_fact_extraction(model, tokenizer, test_examples, rel_descriptions)
    results.update(stage3_results)

    # Compute overall metrics
    overall_accuracy = (
        results['relation_accuracy'] +
        results['head_accuracy'] +
        results['fact_exact_accuracy']
    ) / 3

    results['overall_accuracy'] = overall_accuracy
    results['num_test_examples'] = len(test_examples)

    # Save results
    results_path = Path(model_path) / "autore_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ AutoRE evaluation completed!")
    logger.info(f"📊 Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"📁 Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    # Example usage
    from config import DataConfig

    model_path = "./outputs"  # Path to trained AutoRE model
    test_data_path = "./data/test_processed.json"
    data_config = DataConfig()

    results = evaluate_autore_model(model_path, test_data_path, data_config)
    print(f"AutoRE Evaluation Results: {results}")