---
language:
- kbd
- deu
- eng
- fra
- por
- rus
- spa
- tur
license: cc-by-2.0
task_categories:
- translation
pretty_name: Multilingual-to-Kabardian Tatoeba Translations Dataset
size_categories:
- 1M<n<10M
tags:
- translation
- multilingual
- parallel-corpus
dataset_info:
  features:
  - name: source_tatoeba_id
    dtype: int64
  - name: source_lang
    dtype: string
  - name: source_text
    dtype: string
  - name: target_text
    dtype: string
  - name: similarity_score
    dtype: float64
  - name: nllb_model_version
    dtype: string
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: deu
    num_bytes: 78507756
    num_examples: 530327
  - name: eng
    num_bytes: 343953362
    num_examples: 2252679
  - name: fra
    num_bytes: 72440759
    num_examples: 502436
  - name: por
    num_bytes: 49640754
    num_examples: 340764
  - name: rus
    num_bytes: 156138437
    num_examples: 907621
  - name: spa
    num_bytes: 48255668
    num_examples: 330634
  - name: tur
    num_bytes: 129012850
    num_examples: 868372
  download_size: 278574348
  dataset_size: 877949586
configs:
- config_name: default
  data_files:
  - split: deu
    path: data/deu-*
  - split: eng
    path: data/eng-*
  - split: fra
    path: data/fra-*
  - split: por
    path: data/por-*
  - split: rus
    path: data/rus-*
  - split: spa
    path: data/spa-*
  - split: tur
    path: data/tur-*
---

# Tatoeba Translations Dataset

## Dataset Description

This dataset contains parallel sentence translations from [Tatoeba](https://tatoeba.org/) to Kabardian language (kbd), filtered by similarity score. The source languages are:

- German (deu)
- English (eng)
- French (fra)
- Portuguese (por)
- Russian (rus)
- Spanish (spa)
- Turkish (tur)

All translations in this dataset are paired with Kabardian (kbd) as the target language.

### Dataset Summary

The dataset consists of high-quality parallel translations from multiple languages to Kabardian, containing over 5.7M examples. Each translation pair has been filtered based on similarity scores to ensure quality. This dataset is particularly valuable for multilingual translation tasks involving Kabardian language, as it provides a large-scale resource for low-resource language translation.

### Languages

The dataset includes translations from the following source languages to Kabardian:

Source languages:
- German (deu) [ISO 639-3](https://iso639-3.sil.org/)
- English (eng) [ISO 639-3](https://iso639-3.sil.org/)
- French (fra) [ISO 639-3](https://iso639-3.sil.org/)
- Portuguese (por) [ISO 639-3](https://iso639-3.sil.org/)
- Russian (rus) [ISO 639-3](https://iso639-3.sil.org/)
- Spanish (spa) [ISO 639-3](https://iso639-3.sil.org/)
- Turkish (tur) [ISO 639-3](https://iso639-3.sil.org/)

Target language:
- Kabardian (kbd) [ISO 639-3](https://iso639-3.sil.org/)

## Dataset Structure

### Data Instances

Each instance in the dataset contains:
```python
{
    'source_tatoeba_id': 7627627,                 # Tatoeba sentence ID
    'source_lang': 'eng',                         # Source language code
    'source_text': 'He\'s always happy.',         # Original text
    'target_text': 'Ар сыт щыгъуи насыпыфIэщ.',   # Translated text
    'similarity_score': 0.9283,                   # Translation similarity score
    'nllb_model_version': 'rus-kbd-v1.3.4'        # Translation model version used
}
```

### Data Fields

- `source_tatoeba_id`: Unique identifier from Tatoeba
- `source_lang`: Language code of the source text
- `source_text`: Original text in source language
- `target_text`: Translated text
- `similarity_score`: Similarity score between source and target texts
- `nllb_model_version`: Version of the NLLB model used for translation

### Data Splits

The dataset is organized by source languages, with each language paired with Kabardian as the target language. The similarity thresholds represent the 75th percentile of similarity scores for each language pair:
- deu: German-Kabardian pairs (75th percentile similarity: 0.8711)
- eng: English-Kabardian pairs (75th percentile similarity: 0.8800)
- fra: French-Kabardian pairs (75th percentile similarity: 0.8716)
- por: Portuguese-Kabardian pairs (75th percentile similarity: 0.8807)
- rus: Russian-Kabardian pairs (75th percentile similarity: 0.8916)
- spa: Spanish-Kabardian pairs (75th percentile similarity: 0.8718)
- tur: Turkish-Kabardian pairs (75th percentile similarity: 0.8554)

## Dataset Creation

### Source Data

The source data comes from [Tatoeba](https://tatoeba.org/), a free collaborative online database of example sentences.

#### Initial Data Collection and Normalization

1. Sentences were extracted from Tatoeba's database
2. Filtered by language pairs
3. Translated using various versions of NLLB-200-kbd model
4. Translation quality was assessed using [panagoa/LaBSE-kbd-v0.1](https://huggingface.co/panagoa/LaBSE-kbd-v0.1) model to compute similarity scores
5. For quality assurance, translations were filtered using the 75th percentile of similarity scores for each language pair as a threshold

### Annotations

The translations were generated using different versions of NLLB-200-kbd model. The similarity scores were computed using [panagoa/LaBSE-kbd-v0.1](https://huggingface.co/panagoa/LaBSE-kbd-v0.1) model to ensure translation quality and consistency.

## Additional Information

### Dataset Curators

This dataset was curated by [panagoa]

### Licensing Information

This dataset is licensed under [Creative Commons Attribution 2.0](https://creativecommons.org/licenses/by/2.0/).

### Citation Information

```bibtex
@misc{tatoeba_translations_2025,
    title={Multilingual-to-Kabardian Tatoeba Translations Dataset},
    author={Adam Panagov},
    year={2025},
    publisher={Hugging Face},
    howpublished={\url{https://huggingface.co/datasets/panagoa/tatoeba_kbd}}
}
```

### Contributions

Thanks to [@Tatoeba](https://tatoeba.org/) for providing the source data.

## Usage Examples

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("panagoa/tatoeba_kbd")

# Load specific language pairs
eng_dataset = load_dataset("panagoa/tatoeba_kbd", "eng")
rus_dataset = load_dataset("panagoa/tatoeba_kbd", "rus")

# Example of accessing data
for example in eng_dataset['train']:
    source = example['source_text']
    target = example['target_text']
    similarity = example['similarity_score']
    # Your processing code...
```