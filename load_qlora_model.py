#!/usr/bin/env python3
"""
Helper script to load and use QLoRA fine-tuned models.
"""
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel


def load_qlora_model(adapter_path: str):
    """Load a QLoRA fine-tuned model for inference."""

    print(f"🔄 Loading QLoRA model from: {adapter_path}")

    # Load adapter config to get base model name
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, 'r') as f:
            config = json.load(f)
        base_model_name = config['base_model_name']
        print(f"📋 Base model: {base_model_name}")
    else:
        # Fallback - try to infer from adapter_config.json in the adapter
        try:
            with open(Path(adapter_path) / "adapter_config.json", 'r') as f:
                peft_config = json.load(f)
            base_model_name = peft_config['base_model_name_or_path']
        except:
            raise ValueError("Could not determine base model name. Please specify it manually.")

    # Load tokenizer
    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Setup quantization for inference (optional, for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load base model
    print("🧠 Loading base model with quantization...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load adapter
    print("⚡ Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("✅ QLoRA model loaded successfully!")

    return model, tokenizer


def predict_relation(model, tokenizer, text: str, head_entity: str, tail_entity: str):
    """Predict relation between two entities in text."""

    # Format input
    input_text = f"{text} [SEP] {head_entity} [SEP] {tail_entity}"

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        confidence = torch.softmax(outputs.logits, dim=-1).max().item()

    # Get relation label
    relation = model.config.id2label[predicted_class]

    return relation, confidence


def demo_usage(adapter_path: str):
    """Demo the QLoRA model usage."""

    # Load model
    model, tokenizer = load_qlora_model(adapter_path)

    # Test examples
    test_cases = [
        {
            "text": "Barack Obama was born in Honolulu, Hawaii on August 4, 1961. He later became the 44th President of the United States.",
            "head": "Barack Obama",
            "tail": "Honolulu"
        },
        {
            "text": "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.",
            "head": "Apple Inc.",
            "tail": "Steve Jobs"
        },
        {
            "text": "The University of Oxford is located in Oxford, England and is one of the oldest universities in the world.",
            "head": "University of Oxford",
            "tail": "Oxford"
        }
    ]

    print("\n🎯 Testing QLoRA model predictions:")
    print("="*60)

    for i, case in enumerate(test_cases, 1):
        relation, confidence = predict_relation(
            model, tokenizer,
            case["text"], case["head"], case["tail"]
        )

        print(f"\nTest {i}:")
        print(f"  Text: {case['text'][:60]}...")
        print(f"  {case['head']} → {case['tail']}")
        print(f"  Predicted: {relation} (confidence: {confidence:.3f})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python load_qlora_model.py <adapter_path>")
        print("Example: python load_qlora_model.py ./outputs")
        sys.exit(1)

    adapter_path = sys.argv[1]

    try:
        demo_usage(adapter_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)