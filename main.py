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


def get_user_model_choice():
    print("Welcome to our RE-DocRED LLM Finetuning Tool")
    print("\nChoose a model for relation extraction:")
    print("1. mistralai/Mistral-7B-v0.3")
    print("2. roberta-base")
    print("3. distilbert-base-uncased")
    print("4. microsoft/deberta-v3-base")
    print("5. Custom model from HF Hub (note you may need permissions)")

    while True:
        choice = input("\nSelect a model (1-5) or enter custom model name: ").strip()

        if choice == "1":
            return "mistralai/Mistral-7B-v0.3"
        elif choice == "2":
            return "roberta-base"
        elif choice == "3":
            return "distilbert-base-uncased"
        elif choice == "4":
            return "microsoft/deberta-v3-base"
        elif choice == "5":
            model_name = input("Enter HuggingFace model name: ").strip()
            return model_name
        elif "/" in choice:  # Assume it's a HF model path
            return choice
        else:
            print("Invalid choice. Please try again.")


def setup_directories():
    dirs = ["./data", "./outputs", "./logs"]
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

    # Get model choice
    if args.model:
        model_name = args.model
        print(f"Using specified model: {model_name}")
    else:
        model_name = get_user_model_choice()

    # Initialize configs
    model_config = ModelConfig(model_name=model_name)
    training_config = TrainingConfig(output_dir=args.output_dir)
    data_config = DataConfig(data_dir=args.data_dir)
    hf_config = HuggingFaceConfig()

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

        # Step 2: Training
        print("\n2. Starting model training...")
        from train import train_model

        trainer, train_results = train_model(model_config, training_config, data_config)
        print(f"  ✓ Training completed! Final results: {train_results}")

        # Step 3: Evaluation
        print("\n3. Evaluating model...")
        from evaluate import evaluate_model

        test_data_path = Path(data_config.data_dir) / "test_processed.json"
        if test_data_path.exists():
            eval_results = evaluate_model(training_config.output_dir, str(test_data_path), data_config)
            print(f"    Evaluation completed!")
            print(f"    Accuracy: {eval_results['accuracy']:.4f}")
            print(f"    Macro F1: {eval_results['macro_avg_f1']:.4f}")
            print(f"    Weighted F1: {eval_results['weighted_avg_f1']:.4f}")
        else:
            print("    Test data not found, skipping evaluation...")
            eval_results = None

        # Step 4: HF upload (made optional)
        print("\n4. HuggingFace upload (optional)...")
        if eval_results:
            from upload_to_hf import main_upload_workflow
            main_upload_workflow(training_config.output_dir, model_config.model_name)
        else:
            print("    Skipping upload due to missing eval results")

        print("\nPIPELINE COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"\nFailed with error: {e}")
        raise e


if __name__ == "__main__":
    main()