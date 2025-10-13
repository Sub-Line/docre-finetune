# AutoRE Implementation for RE-DocRED

This repository now implements the **AutoRE RHF (Relation-Head-Facts)** paradigm for document-level relation extraction, based on the research from THUDM.

## 🚀 Major Changes from Classification Approach

### Old Approach (❌ Poor Performance)
- **Model**: `AutoModelForSequenceClassification`
- **Training**: Classification with `[SEP]` tokens
- **Format**: `"text [SEP] head [SEP] tail"` → class_id
- **Problem**: Treated RE as simple classification

### New AutoRE Approach (✅ State-of-the-Art)
- **Model**: `AutoModelForCausalLM` (text generation)
- **Training**: Instruction-following with RHF paradigm
- **Format**: Multi-stage instruction templates
- **Advantage**: Follows AutoRE's proven methodology

## 📁 File Structure

### Core AutoRE Files
- `train_autore.py` - Main AutoRE training script with RHF paradigm
- `evaluate_autore.py` - AutoRE evaluation with 3-stage testing
- `test_autore_model.py` - Interactive testing of trained models
- `supplementary/data_cleaning/relationship_desc.json` - 96 relation descriptions

### Supporting Files
- `main.py` - Updated to use AutoRE training pipeline
- `config.py` - Configuration classes
- `upload_to_hf.py` - HuggingFace model upload

### Legacy Files (Old Classification Approach)
- `train.py` - Old classification training (kept for reference)
- `evaluate.py` - Old classification evaluation
- `load_qlora_model.py` - Old model loading

## 🎯 AutoRE RHF Training Paradigm

The AutoRE approach uses **3-stage instruction following**:

### Stage 1: Relation Extraction
```
Given the following passage, identify all underlying relations:
Passage: {document}
Relations: place_of_birth, founded_by, ...
```

### Stage 2: Head Entity Identification
```
Given the relation "place_of_birth" and passage, identify head entities:
Head entities: Barack Obama, ...
```

### Stage 3: Fact Extraction
```
Given relation "place_of_birth" and head "Barack Obama", extract facts:
Facts: (Barack Obama, place_of_birth, Honolulu)
```

## 🔧 Usage

### Training AutoRE Model
```bash
python main.py --model mistralai/Mistral-7B-v0.3
```

### Testing Trained Model
```bash
python test_autore_model.py ./outputs
```

### Key Features
- ✅ **QLoRA support** for memory-efficient training
- ✅ **96 relation types** with detailed descriptions
- ✅ **Instruction-following** format matching AutoRE paper
- ✅ **3-stage evaluation** (Relation, Head, Fact extraction)
- ✅ **Google Colab compatible**
- ✅ **Auto HuggingFace upload**

## 📊 Expected Performance

AutoRE achieved **state-of-the-art results** on RE-DocRED:
- Surpassed previous methods by **10%+**
- Uses proven RHF paradigm
- Instruction-following approach more suitable for LLMs

## 🎮 Interactive Testing

The `test_autore_model.py` script provides:
- **Demo mode**: Test on predefined examples
- **Interactive mode**: Test your own text
- **3-stage analysis**: See each step of RHF process

## 🔄 Migration from Old Approach

If you have old classification models, they won't work with the new system. You need to retrain using the AutoRE approach for proper performance.

The AutoRE methodology is fundamentally different and much more effective for document-level relation extraction tasks.