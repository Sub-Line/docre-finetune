# RE-DocRED LLM Finetuning Toolkit

Toolkit for finetuning HF large language models on the RE-DocRED dataset for document-level relation extraction and knowledge graph generation.

TASKS
- Downloading and preprocessing RE-DocRED data
- Finetuning any HF model for relation extraction
- Evaluating model performance
- Uploading trained models to HF Hub

converts DocRED relation codes (e.g., P18) to human-readable labels (e.g., "place_of_birth").

```bash
pip install -r requirements.txt
```

```bash
python main.py
```



#### Custom

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

#### Upload to HF

```bash
python upload_to_hf.py
```




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

## Citation

If you use this, please cite the RE-DocRED paper:

```bibtex
@inproceedings{tan-etal-2022-revisiting,
    title = "Revisiting DocRED -- Addressing the False Negative Problem in Relation Extraction",
    author = "Tan, Qingyu and He, Ruidan and Bing, Lidong and Ng, Hwee Tou",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022"
}
```
