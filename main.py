#!/usr/bin/env python3
"""
Main script for finetuning LLMs on RE-DocRED for KG generation.
There is a CLI here just for model selection and training.
"""
import argparse
import os
import sys
from pathlib import Path

from config import ModelConfig, TrainingConfig, DataConfig, HuggingFaceConfig
from huggingface_hub import login, whoami


def get_user_model_choice():
    print("Welcome to our RE-DocRED LLM Finetuning Tool")
    print("\nChoose a model for relation extraction:")
    print("1. mistralai/Mistral-7B-v0.3 (Large, requires good GPU)")
    print("2. microsoft/deberta-v3-large (434M - Excellent for RE)")
    print("3. roberta-large (355M - Great for RE)")
    print("4. microsoft/DialoGPT-medium (345M - Efficient)")
    print("5. roberta-base (125M - Fast baseline)")
    print("6. distilbert-base-uncased (66M - Very fast)")
    print("7. microsoft/deberta-v3-base (184M - Good balance)")
    print("8. Custom model from HF Hub (note you may need permissions)")

    while True:
        choice = input("\nSelect a model (1-8) or enter custom model name: ").strip()

        if choice == "1":
            return "mistralai/Mistral-7B-v0.3"
        elif choice == "2":
            return "microsoft/deberta-v3-large"
        elif choice == "3":
            return "roberta-large"
        elif choice == "4":
            return "microsoft/DialoGPT-medium"
        elif choice == "5":
            return "roberta-base"
        elif choice == "6":
            return "distilbert-base-uncased"
        elif choice == "7":
            return "microsoft/deberta-v3-base"
        elif choice == "8":
            model_name = input("Enter HuggingFace model name: ").strip()
            return model_name
        elif "/" in choice:  # Assume it's a HF model path
            return choice
        else:
            print("Invalid choice. Please try again.")


def setup_hf_authentication():
    """Setup HuggingFace authentication for gated models like Mistral."""
    print("\n🔐 HuggingFace Authentication")
    print("=" * 50)

    # Check if already logged in
    try:
        user_info = whoami()
        print(f"✅ Already logged in as: {user_info['name']}")
        return True
    except Exception:
        pass

    # Check for token in environment
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
            user_info = whoami()
            print(f"✅ Logged in using HF_TOKEN as: {user_info['name']}")
            return True
        except Exception as e:
            print(f"❌ Invalid HF_TOKEN: {e}")

    # Interactive login
    print("🔑 For gated models (like Mistral), you need to login to HuggingFace.")
    print("💡 Get your token from: https://huggingface.co/settings/tokens")

    while True:
        choice = input("\nHow would you like to authenticate?\n1. Enter token now\n2. Skip (for public models only)\nChoice (1/2): ").strip()

        if choice == "1":
            token = input("Enter your HuggingFace token: ").strip()
            if token:
                try:
                    login(token=token)
                    user_info = whoami()
                    print(f"✅ Successfully logged in as: {user_info['name']}")
                    return True
                except Exception as e:
                    print(f"❌ Login failed: {e}")
                    continue
        elif choice == "2":
            print("⚠️  Skipping authentication. Gated models will fail to load.")
            return False
        else:
            print("Please enter 1 or 2.")


def get_upload_approval():
    """Get user pre-approval for HuggingFace upload after training."""
    print("\n📤 HuggingFace Upload Settings")
    print("=" * 50)
    print("Training may take several hours. You can pre-approve model upload")
    print("to HuggingFace Hub so you don't need to monitor the process.")
    print("\nOptions:")
    print("1. Yes - Automatically upload BEST model after successful training")
    print("2. No - Ask me later (requires monitoring)")
    print("3. Skip - Never upload")

    while True:
        choice = input("\nPre-approve HuggingFace upload? (1/2/3): ").strip()

        if choice == "1":
            print("✅ Pre-approved: Will upload BEST model automatically after training")
            return "auto_upload"
        elif choice == "2":
            print("⏰ Will ask for approval after training completes")
            return "ask_later"
        elif choice == "3":
            print("🚫 Upload disabled - model will only be saved locally")
            return "skip_upload"
        else:
            print("Please enter 1, 2, or 3.")


def setup_directories():
    """Create necessary directories and ensure they exist."""
    dirs = ["./data", "./outputs", "./logs", "./backups"]
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")


def main():
    parser = argparse.ArgumentParser(description="Finetune LLM for RE-DocRED")
    parser.add_argument("--model", type=str, help="Model name from HuggingFace Hub")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip RE-DocRED data download")

    args = parser.parse_args()

    # Setup directories
    setup_directories()

    # Setup HuggingFace authentication
    setup_hf_authentication()

    # Get model choice
    if args.model:
        model_name = args.model
        print(f"Using specified model: {model_name}")
    else:
        model_name = get_user_model_choice()

    # Get upload approval before starting training
    upload_approval = get_upload_approval()

    # Initialize configs
    model_config = ModelConfig(model_name=model_name)
    training_config = TrainingConfig(output_dir=args.output_dir)
    data_config = DataConfig(data_dir=args.data_dir)
    hf_config = HuggingFaceConfig()

    # Optimize settings for large models like Mistral
    if 'mistral' in model_name.lower() or '7b' in model_name.lower():
        print("🚀 Detected large model - using AGGRESSIVE QLoRA for maximum efficiency!")
        training_config.batch_size = 1   # Ultra conservative for memory
        training_config.eval_steps = 500  # More reasonable evaluation frequency
        training_config.save_steps = 2000  # Less frequent saving to reduce I/O
        training_config.logging_steps = 25   # Frequent logging to track progress
        data_config.max_examples = 20000  # Restore dataset size for better performance
        print(f"   🎯 AGGRESSIVE QLoRA Optimizations:")
        print(f"   - 4-bit quantization: ~75% memory reduction")
        print(f"   - LoRA adapters: Only train ~0.5% of parameters")
        print(f"   - Ultra-small batch size: {training_config.batch_size}")
        print(f"   - Reduced training examples: {data_config.max_examples}")
        print(f"   - High gradient accumulation for effective batch size")
        print(f"   - Model size on disk: ~15MB (adapters only!)")
        print(f"   ⚡ Memory optimizations: CPU offload + mixed precision")

    print(f"\nConfiguration:")
    print(f"Model: {model_config.model_name}")
    print(f"Data directory: {data_config.data_dir}")
    print(f"Output directory: {training_config.output_dir}")

    # Check if data exists, if not prompt to download
    if not args.skip_download:
        from supplementary.data_cleaning.download_data import check_and_download_data
        check_and_download_data(data_config.data_dir)

    # Start full pipeline
    print("\n" + "="*60)
    print("STARTING RE-DOCRED FINETUNING PIPELINE")
    print("="*60)

    try:
        # Step 1: Data preprocessing
        print("\n1. Preprocessing data...")
        from supplementary.data_cleaning.preprocess_data import preprocess_dataset

        # Process training, dev, and test data
        data_files = [
            ("train_annotated.json", "train_processed.json"),
            ("dev.json", "dev_processed.json"),
            ("test.json", "test_processed.json")
        ]

        rel_info_file = Path(data_config.data_dir) / "rel_info.json"

        for input_file, output_file in data_files:
            input_path = Path(data_config.data_dir) / input_file
            output_path = Path(data_config.data_dir) / output_file

            if input_path.exists() and not output_path.exists():
                preprocess_dataset(str(input_path), str(output_path), str(rel_info_file))
            elif output_path.exists():
                print(f"  ✓ {output_file} already exists")
            else:
                print(f"  ⚠ {input_file} not found, skipping...")

        # Step 2: AutoRE Training
        print("\n2. Starting AutoRE RHF model training...")
        from train_autore import train_autore_model

        trainer, train_results = train_autore_model(model_config, training_config, data_config)
        print(f"  ✓ Training completed! Final results: {train_results}")

        # Step 3: AutoRE Evaluation
        print("\n3. Evaluating AutoRE model...")
        from evaluate_autore import evaluate_autore_model

        test_data_path = Path(data_config.data_dir) / "test_processed.json"
        if test_data_path.exists():
            eval_results = evaluate_autore_model(training_config.output_dir, str(test_data_path), data_config)
            print(f"    AutoRE Evaluation completed!")
            print(f"    Overall Accuracy: {eval_results['overall_accuracy']:.4f}")
            print(f"    Relation Extraction: {eval_results['relation_accuracy']:.4f}")
            print(f"    Head Identification: {eval_results['head_accuracy']:.4f}")
            print(f"    Fact Extraction (Exact): {eval_results['fact_exact_accuracy']:.4f}")
            print(f"    Fact Extraction (Partial): {eval_results['fact_partial_accuracy']:.4f}")
        else:
            print("    Test data not found, skipping evaluation...")
            eval_results = None

        # Step 4: HF upload (based on pre-approval)
        print("\n4. HuggingFace upload...")
        if upload_approval == "skip_upload":
            print("    🚫 Upload skipped by user preference")
        elif not eval_results:
            print("    ⚠️  Skipping upload due to missing eval results")
        elif upload_approval == "auto_upload":
            print("    🚀 Auto-uploading BEST model (pre-approved)...")
            from upload_to_hf import auto_upload_workflow
            auto_upload_workflow(training_config.output_dir, model_config.model_name)
        elif upload_approval == "ask_later":
            print("    ❓ Asking for upload approval...")
            from upload_to_hf import main_upload_workflow
            main_upload_workflow(training_config.output_dir, model_config.model_name)

        print("\nPIPELINE COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"\nFailed with error: {e}")
        raise e


if __name__ == "__main__":
    main()