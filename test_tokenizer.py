#!/usr/bin/env python3
"""
Quick test script to debug tokenizer padding issues with Mistral.
"""
import torch
from transformers import AutoTokenizer

def test_mistral_tokenizer():
    """Test Mistral tokenizer padding."""

    print("🧪 Testing Mistral tokenizer...")

    # Load tokenizer
    model_name = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Original tokenizer state:")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  eos_token: {tokenizer.eos_token}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  unk_token: {tokenizer.unk_token}")
    print(f"  unk_token_id: {tokenizer.unk_token_id}")

    # Apply fix
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # Force EOS for Mistral
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"\nAfter fix:")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")

    # Test batch tokenization
    print(f"\n🔍 Testing batch tokenization...")

    test_texts = [
        "Hello world [SEP] entity1 [SEP] entity2",
        "This is a longer text for testing [SEP] Barack Obama [SEP] United States"
    ]

    try:
        # Test with batch_size > 1
        encoding = tokenizer(
            test_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        print(f"✅ Batch tokenization successful!")
        print(f"  Input IDs shape: {encoding['input_ids'].shape}")
        print(f"  Attention mask shape: {encoding['attention_mask'].shape}")

        # Check for padding tokens
        pad_count = (encoding['input_ids'] == tokenizer.pad_token_id).sum().item()
        print(f"  Padding tokens used: {pad_count}")

    except Exception as e:
        print(f"❌ Batch tokenization failed: {e}")
        return False

    print(f"\n✅ Tokenizer test passed!")
    return True

if __name__ == "__main__":
    test_mistral_tokenizer()