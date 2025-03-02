---
language:
- eng
- kbd
license: cc-by-2.0
task_categories:
- translation
pretty_name: English-to-Kabardian Tatoeba Raw Machine Translations Dataset
size_categories:
- 10M<n<100M
tags:
- translation
- parallel-corpus
dataset_info:
  features:
  - name: tatoeba_id
    dtype: int64
  - name: source_lang
    dtype: string
  - name: target_lang
    dtype: string
  - name: source
    dtype: string
  - name: target
    dtype: string
  - name: similarity
    dtype: float64
  - name: model
    dtype: string
  - name: __index_level_0__
    dtype: int64
  splits:
  - name: train
    num_bytes: 2240011243
    num_examples: 14075008
  download_size: 940834242
  dataset_size: 2240011243
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# English-to-Kabardian Tatoeba Raw Machine Translations Dataset

This dataset contains translations of sentences from English to Kabardian, sourced from the Tatoeba project. It provides a substantial parallel corpus for machine translation and linguistic research involving the Kabardian language.

## Dataset Description

### Dataset Overview

This dataset contains over 14 million sentence pairs, with source sentences in English and their translations in Kabardian. Each entry includes metadata such as the original Tatoeba sentence ID, language codes, and similarity scores.

The dataset serves as a valuable resource for:
- Machine translation systems targeting Kabardian
- Linguistic research on low-resource languages
- Cross-lingual NLP applications
- Preservation and documentation of the Kabardian language

## Dataset Structure

```
{
  'tatoeba_id': 12345,
  'source_lang': 'eng',
  'target_lang': 'kbd',
  'source': 'Example sentence in English',
  'target': 'Translation in Kabardian',
  'similarity': 0.95,
  'model': 'model-name',
  '__index_level_0__': 0
}
```

## Dataset Creation

### Source Data

The source data comes from the [Tatoeba Project](https://tatoeba.org/), which is a collection of sentences and translations in many languages. The English sentences were used as source text for machine translation into Kabardian.

### Translation Process

This dataset is a raw machine translation output without any human validation or post-editing:

- All translations were generated using various fine-tuned versions of the NLLB-200 (No Language Left Behind) model
- No manual corrections or human review was performed on the translations
- The dataset represents direct output from the machine translation system
- The 'similarity' score refers to the model's internal confidence metric
- The 'model' field indicates which specific fine-tuned NLLB-200 version was used for each translation

## Considerations for Using the Data

### Social Impact and Biases

This dataset contributes to language preservation and technological inclusion for the Kabardian language community. However, as with any translation dataset:
- There may be cultural nuances that are difficult to translate accurately
- The distribution of topics may not represent the full breadth of the language
- Potential biases in the source content could be propagated through translations

### Discussion of Limitations

As this is a raw machine translation dataset without human validation:

- Translation quality may vary significantly across the dataset
- There may be systematic errors common to neural machine translation systems
- The translations have not been verified by native Kabardian speakers
- The model may struggle with cultural concepts, idioms, and specialized terminology
- The translations should be considered as a starting point rather than a gold standard
- This dataset is most useful for further fine-tuning of translation models or as a basis for human post-editing

## Additional Information

### Dataset Curators

This dataset has been curated from the Tatoeba project's translations. For specific information about the curation process, please contact the dataset contributors.

### Licensing Information

This dataset is licensed under the Creative Commons Attribution 2.0 Generic (CC BY 2.0) license, which allows for sharing and adaptation with appropriate attribution.

### Citation Information

If you use this dataset in your research, please cite both the Tatoeba project and this dataset collection.

### Contributions

Contributions to improve the dataset are welcome. Please refer to the Hugging Face dataset contribution guidelines.