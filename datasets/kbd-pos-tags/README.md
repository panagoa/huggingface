---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: tagged_text
    dtype: string
  splits:
  - name: train
    num_bytes: 14330108
    num_examples: 82925
  download_size: 7041762
  dataset_size: 14330108
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
language:
- kbd
pretty_name: Kabardian Part-of-Speech Tagging Dataset
task_categories:
- token-classification
- text2text-generation
size_categories:
- 10K<n<100K
---


# Kabardian Part-of-Speech Tagging Dataset

## Dataset Description

This dataset contains Part-of-Speech (POS) annotations for Kabardian (East Circassian) language sentences. The dataset is designed for training and evaluating POS taggers for the Kabardian language.

### Languages
- Kabardian (kbd)

## Dataset Structure

### Format
The dataset follows a token-level annotation format where each token is labeled with its corresponding POS tag using XML-style tags.

Example:
```
Абы<PRON> и<PRON> Iуэхум<NOUN> сэ<PRON> нэхъыбэ<ADV> зэрыхэсщIыкIраи<VERB> сызыукIыр<VERB>
```

### POS Tags
The dataset uses Universal POS tags:
- `ADJ`: adjectives
- `ADP`: adpositions
- `ADV`: adverbs
- `AUX`: auxiliaries
- `CCONJ`: coordinating conjunctions
- `DET`: determiners
- `INTJ`: interjections
- `NOUN`: nouns
- `NUM`: numerals
- `PART`: particles
- `PRON`: pronouns
- `PROPN`: proper nouns
- `PUNCT`: punctuation
- `SCONJ`: subordinating conjunctions
- `SYM`: symbols
- `VERB`: verbs
- `X`: other

## Uses and Limitations

### Intended Uses
- Training POS taggers for Kabardian
- Linguistic research on Caucasian languages
- Development of NLP tools for Kabardian
- Comparative linguistic studies

### Limitations
- Limited coverage of linguistic phenomena
- May not cover all dialectal variations
- [Add specific limitations]

## Additional Information

### Annotation Guidelines
Each token is annotated with one of the predefined POS tags. The annotation follows these principles:
1. Tokens maintain their original orthographic form
2. Each token receives exactly one tag
3. Tags are assigned based on the token's function in the sentence

### Statistics
[Add dataset statistics:
- Number of sentences
- Number of tokens
- Distribution of POS tags]

### Quality Control
The dataset was created using a zero-shot POS tagging approach with the Gemini 2.0 Flash language model. The annotation process included:

1. A detailed prompt with:
   - Complete Universal POS tags inventory
   - Annotation guidelines
   - Example annotations
   - Special cases handling instructions

2. The annotation process followed these steps:
   - Basic POS identification
   - Context-based tag refinement
   - Special constructions verification

3. Technical details:
   - Batch processing with size of 40 sentences per request
   - Temperature setting of 0.1 for consistent outputs
   - Systematic logging of all model outputs

## Source Data
[Add information about the source of the sentences]

## Dataset Creator
[Add creator information]

## License
[Add license information]

## Citation
[Add citation information if applicable]