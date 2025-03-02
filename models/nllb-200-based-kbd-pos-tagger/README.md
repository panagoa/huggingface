---
library_name: transformers
language:
- kbd
base_model:
- panagoa/nllb-200-distilled-600M-kbd-v0.1
---

# NLLB-200 Based Kabardian POS Tagger

This is a fine-tuned version of [panagoa/nllb-200-1.3b-kbd-v0.1](https://huggingface.co/panagoa/nllb-200-1.3b-kbd-v0.1) for Part-of-Speech (POS) tagging of Kabardian (East Circassian) language. The model was trained on the [Kabardian Part-of-Speech Tagging Dataset](https://huggingface.co/datasets/panagoa/kbd-pos-tags).

## Model Description

The model uses NLLB-200's encoder-decoder architecture to perform sequence-to-sequence POS tagging, where input is a Kabardian sentence and output is the same sentence with Universal POS tags.

### Input Format
Plain Kabardian text, for example:
```
Абы и Iуэхум сэ нэхъыбэ зэрыхэсщIыкIраи сызыукIыр
```

### Output Format
Text with XML-style POS tags:
```
Абы<PRON> и<PRON> Iуэхум<NOUN> сэ<PRON> нэхъыбэ<ADV> зэрыхэсщIыкIраи<VERB> сызыукIыр<VERB>
```

### Universal POS Tags Used
The model uses the Universal POS tagset:
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

## Usage

```python
from transformers import AutoTokenizer, M2M100ForConditionalGeneration

# Load model and tokenizer
model_name = "panagoa/nllb-200-based-kbd-pos-tagger"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Function for POS tagging
def pos_tag_sentence(text, model, tokenizer, output_format='mixed', max_length=128):
    """
    Perform POS tagging on a Kabardian sentence.
    Args:
        text (str): Input sentence to tag
        model: The trained model
        tokenizer: The tokenizer
        output_format: One of 'mixed' (слово<TAG>), 'words' (только слова),
                      'tags' (только теги), or 'pairs' (список пар (слово, тег))
        max_length (int): Maximum sequence length
    Returns:
        Based on output_format:
        - 'mixed': str with format "слово<TAG> слово<TAG>"
        - 'words': list of words
        - 'tags': list of tags
        - 'pairs': list of (word, tag) tuples
    """
    inputs = tokenizer(text,
                      return_tensors="pt",
                      max_length=max_length,
                      truncation=True)

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        num_beams=2,
        do_sample=False,
        # temperature=0.7,
        # top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True,
        max_new_tokens=128,
    )

    # Получаем предсказание модели
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Парсим результат
    tagged_tokens = result.strip().split()
    words = []
    tags = []

    for token in tagged_tokens:
        match = re.match(r'(.+)<([A-Z]+)>', token)
        if match:
            word, tag = match.groups()
            words.append(word)
            tags.append(tag)

    # Возвращаем результат в запрошенном формате
    if output_format == 'mixed':
        return result
    elif output_format == 'words':
        return words
    elif output_format == 'tags':
        return tags
    elif output_format == 'pairs':
        return list(zip(words, tags))
    else:
        raise ValueError(f"Unknown output format: {output_format}")

# Example usage
print(pos_tag_sentence('Шкафым фалъэр дэкъутыхьащ.', model, tokenizer, output_format='mixed'))
</s>kbd_Cyrl Шкафым<NOUN> фалъэр<NOUN> дэкъутыхьащ<VERB>.</s>

print(pos_tag_sentence('Шкафым фалъэр дэкъутыхьащ.', model, tokenizer, output_format='pairs'))
[('Шкафым', 'NOUN'), ('фалъэр', 'NOUN'), ('дэкъутыхьащ', 'VERB')]
```

## Training

The model was trained on the [Kabardian Part-of-Speech Tagging Dataset](https://huggingface.co/datasets/panagoa/kbd-pos-tags) which contains 82,925 annotated sentences. The training used the following configuration:

- Learning rate: 2e-5
- Batch size: 16
- Training epochs: 3
- Max sequence length: 256
- Optimizer: AdamW with weight decay 0.01

## Evaluation

The model was evaluated on a held-out test set with the following metrics:
- Tag accuracy: How often individual POS tags are correct
- Sentence accuracy: How often entire sentences are correctly tagged

## Limitations

- The model's performance may vary for:
  - Very long sentences
  - Dialectal variations
  - Non-standard orthography
  - Specialized or technical vocabulary
- The training data was created using a zero-shot approach, which may introduce some biases

## Intended Use

This model is intended for:
- POS tagging of Kabardian text
- Linguistic research on Kabardian
- Development of NLP tools for Kabardian
- Comparative linguistic studies

## Citation

If you use this model, please cite both the model and the dataset:

```bibtex
@misc{kbd-pos-tagger,
  author = {Panagoa},
  title = {NLLB-200 Based Kabardian POS Tagger},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/panagoa/nllb-200-based-kbd-pos-tagger}}
}
```

## License

This model inherits the license of the base NLLB-200 model and the dataset used for fine-tuning.