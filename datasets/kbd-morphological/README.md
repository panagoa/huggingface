---
dataset_info:
  features:
  - name: word
    dtype: string
  - name: gloss
    dtype: string
  - name: morphological_tags
    dtype: string
  - name: translation
    dtype: string
  splits:
  - name: train
    num_bytes: 3612465
    num_examples: 33905
  download_size: 1723271
  dataset_size: 3612465
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
language:
- kbd
pretty_name: Kabardian Morphological
task_categories:
- token-classification
size_categories:
- 10K<n<100K
---


# Kabardian Morphological Analysis Dataset

## Dataset Description

This dataset contains morphological annotations for Kabardian (East Circassian) language words, including morpheme segmentation, grammatical features, and translations.

### Languages
- Kabardian (kbd)
- Russian (rus) - translations

## Dataset Structure

The dataset contains the following columns:
- `word`: Original word in Kabardian
- `gloss`: Morphological segmentation with hyphens
- `morphological_tags`: Grammatical features (e.g., 3pl-buy-past-aff)
- `translation`: Russian translation

Example:
```
word: ящэхуащ
gloss: я-щэху-а-щ
morphological_tags: 3pl-buy-past-aff
translation: "они купили"
```

## Uses and Limitations

### Intended Uses
- Morphological analysis of Kabardian
- Training morphological segmentation models
- Linguistic research on Caucasian languages
- Development of NLP tools for low-resource languages
- Research on agglutinative morphology

### Limitations
- Limited to verbal morphology
- Translations provided only in Russian
- Dataset size may not be sufficient for large-scale machine learning
- May not cover all dialectal variations

## Additional Information

The Kabardian language belongs to the Northwest Caucasian family and is known for its complex morphological system, particularly in verbal forms. This dataset focuses on capturing the rich morphological structure of Kabardian verbs, which can encode various grammatical categories including person, number, tense, aspect, mood, and valency through agglutinative morphology.

### Features
- Detailed morphological segmentation
- Morphological glossing with consistent internal annotation scheme
- Russian translations for semantic clarity
- Focus on verbal morphology

### Source Data
Todo

### Data Collection Process
The dataset was created using a zero-shot morphological analysis approach with the Claude 3.5 Sonnet language model. The annotation process included:

1. A detailed prompt engineering system with:
   - Comprehensive morpheme inventory
   - Glossing conventions
   - Annotation guidelines
   - Example analyses
   - Full list of grammatical features
   
2. Batch processing with size of 40 words per request

The annotations were produced using a custom prompting system that guided the model in consistent morphological analysis of Kabardian words.

## Dataset Creator
Todo

## License
Todo