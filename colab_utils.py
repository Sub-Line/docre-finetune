"""
Utility functions for Google Colab to handle disconnections and downloads.
"""
import os
import zipfile
from pathlib import Path
from google.colab import files
import shutil


def download_model_files():
    """Download trained model files from Colab to local machine."""
    print("📦 Preparing model files for download...")

    # Check what models are available
    models_to_download = []

    # Check main outputs
    if Path("./outputs").exists():
        models_to_download.append("./outputs")

    # Check backups
    backup_dirs = list(Path("./backups").glob("model_*"))
    if backup_dirs:
        latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)
        models_to_download.append(str(latest_backup))

    if not models_to_download:
        print("❌ No trained models found to download!")
        return

    # Create zip file with all models
    zip_filename = "re_docred_models.zip"

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for model_dir in models_to_download:
            model_path = Path(model_dir)
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    arcname = str(file_path.relative_to(Path(".")))
                    zipf.write(file_path, arcname)

    print(f"✅ Created {zip_filename} with trained models")

    # Download training and evaluation summaries if they exist
    summary_files = ["training_summary.txt", "evaluation_summary.txt"]
    for summary_file in summary_files:
        if Path(summary_file).exists():
            files.download(summary_file)

    # Download the main zip file
    files.download(zip_filename)

    print("🎉 Download complete!")
    print("Extract the zip file locally to get your trained models.")


def setup_colab_environment():
    """Setup environment variables and configurations for Colab."""
    print("🚀 Setting up Colab environment for RE-DocRED training...")

    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected. Training will be very slow on CPU.")

    # Set up directories
    dirs = ["./data", "./outputs", "./logs", "./backups"]
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)

    print("✅ Directories created")

    # Optimized settings for Colab
    colab_recommendations = """
📋 Colab Training Recommendations:

🔧 For faster training:
- Use batch_size=16 (if you have enough GPU memory)
- Reduce num_epochs=2 for initial testing
- Use distilbert-base-uncased for faster training

💾 To prevent data loss:
- Models are automatically backed up with timestamps
- Download your models before the session ends
- Use the download_model_files() function

⚡ If you get disconnected:
- Your models are saved in ./outputs and ./backups
- Re-run the evaluation script to check results
- Use the backup models if main outputs are corrupted

🔑 For Mistral models:
- Make sure you're logged into HuggingFace
- You may need to request access to gated models first
"""
    print(colab_recommendations)


def check_training_status():
    """Check the status of training and available models."""
    print("🔍 Checking training status...")

    # Check if training is in progress
    if Path("./outputs/training_args.bin").exists():
        print("🔄 Training appears to be in progress or completed")

    # Check main outputs
    if Path("./outputs").exists():
        output_files = list(Path("./outputs").glob("*"))
        print(f"📁 Main outputs directory: {len(output_files)} files")

        # Check for key files
        key_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "label_mapping.json"]
        for key_file in key_files:
            if (Path("./outputs") / key_file).exists():
                print(f"   ✅ {key_file}")
            else:
                print(f"   ❌ {key_file}")

    # Check backups
    backup_dirs = list(Path("./backups").glob("model_*"))
    if backup_dirs:
        print(f"💾 Backup models: {len(backup_dirs)} available")
        latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)
        print(f"   Latest: {latest_backup.name}")
    else:
        print("💾 No backup models found")

    # Check summaries
    summaries = ["training_summary.txt", "evaluation_summary.txt"]
    for summary in summaries:
        if Path(summary).exists():
            print(f"📄 {summary} available")
            # Show first few lines
            with open(summary, 'r') as f:
                lines = f.readlines()[:5]
                for line in lines:
                    print(f"     {line.strip()}")
            print("     ...")


def quick_test_model():
    """Quick test of a trained model with a sample input."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_path = "./outputs"
    if not Path(model_path).exists():
        # Try backup
        backup_dirs = list(Path("./backups").glob("model_*"))
        if backup_dirs:
            model_path = str(max(backup_dirs, key=lambda x: x.stat().st_mtime))
        else:
            print("❌ No trained model found!")
            return

    print(f"🧪 Testing model from: {model_path}")

    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Test input
        text = "Barack Obama was born in Honolulu, Hawaii. He became the 44th President of the United States."
        head_entity = "Barack Obama"
        tail_entity = "Honolulu"

        # Predict
        input_text = f"{text} [SEP] {head_entity} [SEP] {tail_entity}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            confidence = torch.softmax(outputs.logits, dim=-1).max().item()

        relation = model.config.id2label[predicted_class]

        print(f"✅ Model test successful!")
        print(f"   Input: {head_entity} -> {tail_entity}")
        print(f"   Predicted relation: {relation}")
        print(f"   Confidence: {confidence:.4f}")

    except Exception as e:
        print(f"❌ Model test failed: {e}")


if __name__ == "__main__":
    print("🔧 Colab Utilities for RE-DocRED Training")
    print("Available functions:")
    print("- setup_colab_environment()")
    print("- check_training_status()")
    print("- quick_test_model()")
    print("- download_model_files()")