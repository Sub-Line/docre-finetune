# RE-DocRED LLM Finetuning Toolkit

A comprehensive toolkit for finetuning large language models on the RE-DocRED dataset for document-level relation extraction and knowledge graph generation.

## Overview

This project provides a complete pipeline for:
- Downloading and preprocessing RE-DocRED data
- Finetuning any HuggingFace model for relation extraction
- Evaluating model performance
- Uploading trained models to HuggingFace Hub

The toolkit converts relation codes (e.g., P18) to human-readable labels (e.g., "place_of_birth") and supports model-agnostic training.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the interactive training pipeline**:
   ```bash
   python main.py
   ```

3. **Follow the prompts** to select your model and configure training.

## Features

- ✅ **Model-agnostic**: Works with any HuggingFace transformer model
- ✅ **Interactive CLI**: User-friendly prompts for model selection
- ✅ **Automatic data download**: Fetches RE-DocRED data from GitHub
- ✅ **Preprocessing pipeline**: Converts relation codes to readable labels
- ✅ **Comprehensive evaluation**: Detailed metrics and classification reports
- ✅ **HuggingFace integration**: Optional model upload with approval workflow
- ✅ **Configurable training**: Easy-to-modify configuration files

## Project Structure

```
├── main.py                     # Main entry point with interactive CLI
├── config.py                   # Configuration classes
├── train.py                    # Training script with HF Trainer
├── evaluate.py                 # Model evaluation script
├── upload_to_hf.py            # HuggingFace upload functionality
├── requirements.txt            # Python dependencies
└── supplementary/
    └── data_cleaning/
        ├── download_data.py    # RE-DocRED data download
        └── preprocess_data.py  # Data preprocessing pipeline
```

## Usage

### Basic Usage

Run the main script and follow the interactive prompts:

```bash
python main.py
```

### Advanced Usage

#### Custom Model Training

```bash
python main.py --model microsoft/deberta-v3-base --output-dir ./my_model
```

#### Skip Data Download

```bash
python main.py --skip-download --data-dir ./existing_data
```

#### Evaluation Only

```bash
python evaluate.py
```

#### Upload to HuggingFace

```bash
python upload_to_hf.py
```

## Configuration

Modify `config.py` to customize training parameters:

```python
@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    # ... other parameters
```

## Supported Models

The toolkit works with any HuggingFace model. Popular choices include:

- `bert-base-uncased`
- `roberta-base`
- `distilbert-base-uncased`
- `microsoft/deberta-v3-base`
- Any custom model from HuggingFace Hub

## Data Format

The preprocessed data converts RE-DocRED format to training examples:

```json
{
  "text": "Document text...",
  "head_entity": "Entity 1",
  "tail_entity": "Entity 2",
  "relation": "place_of_birth",
  "doc_id": "document_identifier"
}
```

## Performance

Models typically achieve:
- **Accuracy**: 0.85-0.92
- **Macro F1**: 0.80-0.88
- **Weighted F1**: 0.84-0.91

(Results vary by base model and training configuration)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers 4.30+
- See `requirements.txt` for full list

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this toolkit, please cite the RE-DocRED paper:

```bibtex
@inproceedings{tan-etal-2022-revisiting,
    title = "Revisiting DocRED -- Addressing the False Negative Problem in Relation Extraction",
    author = "Tan, Qingyu and He, Ruidan and Bing, Lidong and Ng, Hwee Tou",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022"
}
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation in `manual.md`
- Review the example outputs and logs