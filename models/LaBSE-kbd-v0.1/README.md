---
language:
- multilingual
- kbd
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
library_name: sentence-transformers
license: apache-2.0
base_model:
- sentence-transformers/LaBSE
---

# LaBSE
This is a port of the [LaBSE](https://tfhub.dev/google/LaBSE/1) model to PyTorch. It can be used to map 109 languages to a shared vector space.


## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('panagoa/LaBSE-kbd-v0.1')

rus_text = "Не беспокойся."
kbd_text = "Умыгузавэ."

embeddings = model.encode([rus_text, kbd_text])

similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity: {similarity:.4f}")

Similarity: 0.9194
```



## Evaluation Results



For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name=sentence-transformers/LaBSE)



## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  (2): Dense({'in_features': 768, 'out_features': 768, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
  (3): Normalize()
)
```

## Citing & Authors

Have a look at [LaBSE](https://tfhub.dev/google/LaBSE/1) for the respective publication that describes LaBSE.