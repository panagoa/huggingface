---
library_name: transformers
base_model:
- panagoa/xlm-roberta-base-kbd
language:
- kbd
tags:
- Part-of-Speech
- XLM-RoBERTa
datasets:
- panagoa/kbd-pos-tags
pipeline_tag: token-classification
---


# XLM-RoBERTa for Kabardian Part-of-Speech Tagging

## Model description

This model is a fine-tuned version of [panagoa/xlm-roberta-base-kbd](https://huggingface.co/panagoa/xlm-roberta-base-kbd) on the [panagoa/kbd-pos-tags](https://huggingface.co/datasets/panagoa/kbd-pos-tags) dataset. It is designed to perform Part-of-Speech (POS) tagging for text in the Kabardian language (kbd).

The model identifies 17 different POS tags:

| Tag | Description | Examples |
|-----|-------------|----------|
| ADJ | Adjective | хужь (white), къабзэ (clean) |
| ADP | Adposition | щхьэкIэ (for), папщIэ (because of) |
| ADV | Adverb | псынщIэу (quickly), жыжьэу (far) |
| AUX | Auxiliary | хъунщ (will be), щытащ (was) |
| CCONJ | Coordinating conjunction | икIи (and), ауэ (but) |
| DET | Determiner | мо (that), мыпхуэдэ (this kind) |
| INTJ | Interjection | уэлэхьи (by God), зиунагъуэрэ (oh my) |
| NOUN | Noun | унэ (house), щIалэ (boy) |
| NUM | Numeral | зы (one), тIу (two) |
| PART | Particle | мы (this), а (that) |
| PRON | Pronoun | сэ (I), уэ (you) |
| PROPN | Proper noun | Мурат (Murat), Налшык (Nalchik) |
| PUNCT | Punctuation | . (period), , (comma) |
| SCONJ | Subordinating conjunction | щхьэкIэ (because), щыгъуэ (when) |
| SYM | Symbol | % (percent), $ (dollar) |
| VERB | Verb | мэкIуэ (goes), матхэ (writes) |
| X | Other | - |


## Intended Use

This model is intended for:
- Linguistic analysis of Kabardian text
- Natural language processing pipelines for Kabardian
- Research on low-resource languages
- Educational purposes for teaching Kabardian grammar

## Training Data

The model was trained on the [panagoa/kbd-pos-tags](https://huggingface.co/datasets/panagoa/kbd-pos-tags) dataset, which contains 82,925 tagged sentences in Kabardian. The dataset shows the following tag distribution:

- VERB: 116,377 (30.0%)
- NOUN: 115,232 (29.7%)
- PRON: 63,827 (16.5%)
- ADV: 35,036 (9.0%)
- ADJ: 20,817 (5.4%)
- PROPN: 18,692 (4.8%)
- DET: 6,830 (1.8%)
- CCONJ: 6,098 (1.6%)
- ADP: 4,793 (1.2%)
- PUNCT: 4,752 (1.2%)
- NUM: 4,741 (1.2%)
- INTJ: 2,787 (0.7%)
- PART: 2,241 (0.6%)
- SCONJ: 1,206 (0.3%)
- AUX: 560 (0.1%)
- X: 273 (0.1%)
- SYM: 7 (<0.1%)

## Training Procedure

The model was trained with the following configuration:
- Base model: panagoa/xlm-roberta-base-kbd
- Learning rate: 2e-5
- Batch size: 32
- Epochs: 3
- Weight decay: 0.01
- Class weights: Applied to handle class imbalance
- Maximum sequence length: 128

Class weights were calculated inversely proportional to the class frequencies to address the imbalance in the dataset, with rare tags given higher importance during training.

## Evaluation Results

The model achieved the following performance on a validation set (20% of the data):
- Overall accuracy: ~85%
- Performance varies across different POS tags, with better results on common tags like NOUN and VERB.

## Limitations

- The model may struggle with rare POS tags (like SYM) due to limited examples in the training data
- Performance may vary with dialectal variations or non-standard Kabardian text
- The model has a context window limitation of 128 tokens
- Some ambiguous words might be incorrectly tagged based on context

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("panagoa/xlm-roberta-base-kbd-pos-tagger")
model = AutoModelForTokenClassification.from_pretrained("panagoa/xlm-roberta-base-kbd-pos-tagger")

# Define function for prediction
def predict_pos_tags(text, model, tokenizer):
    # Split text into words if it's a string
    if isinstance(text, str):
        text = text.split()
        
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Tokenize input text
    encoded_input = tokenizer(
        text,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt"
    )
    
    # Move inputs to the same device
    inputs = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Map to POS tags
    word_ids = encoded_input.word_ids()
    previous_word_idx = None
    predicted_tags = []
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx != previous_word_idx:
            predicted_tags.append(model.config.id2label[predictions[0][idx].item()])
        previous_word_idx = word_idx
    
    return predicted_tags[:len(text)]

# Example usage
text = "Хъыджэбзыр щIэкIри фошыгъу къыхуихьащ"
words = text.split()
tags = predict_pos_tags(words, model, tokenizer)

# Print results
for word, tag in zip(words, tags):
    print(f"{word}: {tag}")

Хъыджэбзыр: NOUN
щIэкIри: VERB
фошыгъу: NOUN
къыхуихьащ: VERB
```

## Author

This model was trained by panagoa and contributed to the Hugging Face community to support NLP research and applications for the Kabardian language.

## Citation

If you use this model in your research, please cite:

```
@misc{panagoa2025kabardianpos,
  author = {Panagoa},
  title = {XLM-RoBERTa for Kabardian Part-of-Speech Tagging},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/panagoa/xlm-roberta-base-kbd-pos-tagger}}
}
```