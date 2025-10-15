"""
Testing script for AutoRE models - load and test trained models.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json

def load_autore_model(model_path: str):
    """Load AutoRE model for testing."""

    print(f"🔄 Loading AutoRE model from: {model_path}")

    try:
        # Try loading as PEFT model first
        try:
            # First check if PEFT config file exists and is valid
            adapter_config_path = f"{model_path}/adapter_config.json"
            try:
                with open(adapter_config_path, 'r') as f:
                    config_data = json.load(f)
                print(f"📄 Found adapter config: {config_data}")
            except FileNotFoundError:
                print("📄 No adapter_config.json found")

            peft_config = PeftConfig.from_pretrained(model_path)
            print("📦 Detected QLoRA adapter")

            # Load base model with quantization
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )

            # Load adapter
            model = PeftModel.from_pretrained(base_model, model_path)
            print("✅ QLoRA model loaded successfully!")

        except Exception as e:
            print(f"PEFT loading failed: {e}")
            print("📦 Loading as regular CausalLM model")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("✅ Regular model loaded successfully!")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 100):
    """Generate response from AutoRE model."""

    # Tokenize
    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate
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

    # Decode response (remove input part)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(instruction):].strip()

    return response


def test_relation_extraction(model, tokenizer, text: str):
    """Test Stage 1: Relation Extraction."""

    instruction = f"""Given the following passage, identify all underlying relations that exist between entities:

Passage: {text}

Instructions: List any relations that you can identify in this passage. Focus on factual relationships between entities mentioned in the text.

Relations:"""

    response = generate_response(model, tokenizer, instruction, max_new_tokens=100)
    return response


def test_head_identification(model, tokenizer, text: str, relation: str):
    """Test Stage 2: Head Entity Identification."""

    instruction = f"""Given the relation "{relation}" and the following passage, identify all entities that could serve as the subject (head entity) for this relation:

Relation: {relation}

Passage: {text}

Instructions: List entities that can be the subject of the "{relation}" relation.

Head entities:"""

    response = generate_response(model, tokenizer, instruction, max_new_tokens=100)
    return response


def test_fact_extraction(model, tokenizer, text: str, relation: str, head_entity: str):
    """Test Stage 3: Fact Extraction."""

    instruction = f"""Given the relation "{relation}", the head entity "{head_entity}", and the following passage, extract the complete triplet facts:

Relation: {relation}
Head Entity: {head_entity}

Passage: {text}

Instructions: List all complete triplet facts in the format (head, relation, tail) that involve the specified relation and head entity.

Facts:"""

    response = generate_response(model, tokenizer, instruction, max_new_tokens=150)
    return response


def demo_autore_model(model_path: str):
    """Demo the AutoRE model with test cases."""

    # Load model
    model, tokenizer = load_autore_model(model_path)
    if model is None:
        return

    print("\n🎯 Testing AutoRE Model Performance:")
    print("=" * 60)

    # Test cases
    test_cases = [
        {
            "text": "Barack Obama was born in Honolulu, Hawaii on August 4, 1961. He later served as the 44th President of the United States from 2009 to 2017.",
            "expected_relation": "place_of_birth",
            "expected_head": "Barack Obama",
            "expected_tail": "Honolulu"
        },
        {
            "text": "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in Cupertino, California.",
            "expected_relation": "founded_by",
            "expected_head": "Apple Inc.",
            "expected_tail": "Steve Jobs"
        },
        {
            "text": "Microsoft Corporation is headquartered in Redmond, Washington. The company was founded in 1975 by Bill Gates and Paul Allen.",
            "expected_relation": "headquarters_location",
            "expected_head": "Microsoft Corporation",
            "expected_tail": "Redmond"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Text: {case['text']}")
        print(f"Expected: {case['expected_head']} → {case['expected_relation']} → {case['expected_tail']}")
        print()

        # Stage 1: Relation Extraction
        relations = test_relation_extraction(model, tokenizer, case['text'])
        print(f"🔍 Stage 1 - Relations Found: {relations}")

        # Stage 2: Head Entity Identification
        heads = test_head_identification(model, tokenizer, case['text'], case['expected_relation'])
        print(f"🎯 Stage 2 - Head Entities: {heads}")

        # Stage 3: Fact Extraction
        facts = test_fact_extraction(model, tokenizer, case['text'], case['expected_relation'], case['expected_head'])
        print(f"📋 Stage 3 - Facts Extracted: {facts}")

        # Simple evaluation
        success_indicators = [
            case['expected_relation'].lower() in relations.lower(),
            case['expected_head'].lower() in heads.lower(),
            case['expected_tail'].lower() in facts.lower()
        ]

        success_rate = sum(success_indicators) / len(success_indicators)

        if success_rate >= 0.67:
            print(f"✅ Test {i}: SUCCESS ({success_rate:.2%} correct)")
        elif success_rate >= 0.33:
            print(f"⚠️ Test {i}: PARTIAL ({success_rate:.2%} correct)")
        else:
            print(f"❌ Test {i}: FAILED ({success_rate:.2%} correct)")

    print(f"\n🎉 AutoRE model testing complete!")


def interactive_test(model_path: str):
    """Interactive testing of AutoRE model."""

    model, tokenizer = load_autore_model(model_path)
    if model is None:
        return

    print("\n🎮 Interactive AutoRE Testing")
    print("Enter 'quit' to stop")

    while True:
        try:
            print("\n" + "-"*50)
            text = input("📝 Enter text: ").strip()
            if text.lower() == 'quit':
                break

            print(f"\n🔍 Analyzing: {text[:100]}...")

            # Stage 1: Find relations
            relations = test_relation_extraction(model, tokenizer, text)
            print(f"\n📊 Relations found: {relations}")

            # Ask user which relation to explore
            relation = input("\n🎯 Enter relation to explore: ").strip()
            if not relation:
                continue

            # Stage 2: Find head entities
            heads = test_head_identification(model, tokenizer, text, relation)
            print(f"\n🎯 Head entities: {heads}")

            # Ask user for head entity
            head_entity = input("\n🎯 Enter head entity: ").strip()
            if not head_entity:
                continue

            # Stage 3: Extract facts
            facts = test_fact_extraction(model, tokenizer, text, relation, head_entity)
            print(f"\n📋 Facts extracted: {facts}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n👋 Thanks for testing AutoRE!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python test_autore_model.py <model_path>")
        print("Example: python test_autore_model.py ./outputs")
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        print("🚀 AutoRE Model Testing")
        print("=" * 40)

        # Run demo first
        demo_autore_model(model_path)

        # Offer interactive testing
        if input("\n🎮 Run interactive testing? (y/n): ").lower().startswith('y'):
            interactive_test(model_path)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)