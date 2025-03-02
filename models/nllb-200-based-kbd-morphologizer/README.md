---
library_name: transformers
language:
- kbd
base_model:
- panagoa/nllb-200-distilled-600M-kbd-v0.1
pipeline_tag: text2text-generation
---

```python
def word_to_morph_features(text):
    prefix = "<word analyze>: "
    inputs = tokenizer(prefix + text, return_tensors="pt", max_length=128, truncation=True)

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


test_words = [
  'зезгъэщхьырт',
  'псыхьыжауэ',
  'хъужами',
  'япыщIат',
  'къыщежьэри',
  'схъумэрт',
  'щхьэпэщ',
  'сылъакъуэ',
  'къеджэмэ',
  'гъэщам',
  'бэракъыу',
  'уеупщIакъым',
  'къэзыгъэпэж',
  'къахуэбла',
  'иращIэнтэкъым',
  'къыбоух',
  'гъусари',
  'сщитIэгъащ',
  'дытелажьэу',
]
for word in test_words:
    features = word_to_morph_features(word)
    print(f"{word} -> {features}")
```

```text
зезгъэщхьырт -> <features>: 1sg-pre-1sg-caus-know-past
псыхьыжауэ -> <features>: water-water-adv
хъужами -> <features>: become-past-conn
япыщIат -> <features>: 3pl-attach-past-aff
къыщежьэри -> <features>: hor-begin-and
схъумэрт -> <features>: 1sg-caus-stand-epv-fut
щхьэпэщ -> <features>: dir-ben-val-aff
сылъакъуэ -> <features>: 1sg-run-adv
къеджэмэ -> <features>: hor-read-cond
гъэщам -> <features>: year-erg
бэракъыу -> <features>: flag-adv
уеупщIакъым -> <features>: 2sg-ben-ask-neg
къэзыгъэпэж -> <features>: hor-rel-caus-caus-run-back
къахуэбла -> <features>: hor-ben-loc-approach
иращIэнтэкъым -> <features>: 3pl-ben-3pl-do-epv-fut-neg
къыбоух -> <features>: 2sg-dir-2sg-caus-fall
гъусари -> <features>: companion-and
сщитIэгъащ -> <features>: 1sg-ben-1sg-put-past
дытелажьэу -> <features>: 1pl-dir-work-adv
```