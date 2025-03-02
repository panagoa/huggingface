---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:3395988
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/LaBSE
widget:
- source_sentence: Tom grabbed Mary's elbow.
  sentences:
  - Tom, Mary'yi dirseğinden kavradı.
  - Stay with her.
  - Pourquoi a-t-il mangé l'abeille ?
- source_sentence: Жизнь - это тень.
  sentences:
  - Life is a shadow.
  - I'm almost always at home on Sundays.
  - Henüz bir vizem yok.
- source_sentence: Are you working tomorrow?
  sentences:
  - Yarın çalışacak mısın?
  - Нобэ хуабей дыдэт.
  - Мэри къэшэн имыIэну жеIэ.
- source_sentence: Вы нарушили закон.
  sentences:
  - Ахэр Iейщ.
  - Tom war drei Tage nicht da.
  - Vous avez enfreint la loi.
- source_sentence: We've never seen Tom this angry before.
  sentences:
  - Tom'u daha önce asla bu kadar öfkeli görmedik.
  - Soyez attentive aux voleurs à la tire.
  - Endişeli görünüyorsun.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/LaBSE
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: validation
      type: validation
    metrics:
    - type: pearson_cosine
      value: -0.2799955028525394
      name: Pearson Cosine
    - type: spearman_cosine
      value: -0.32115994644018286
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/LaBSE

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE) <!-- at revision b7f947194ceae0ddf90bafe213722569e274ad28 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Dense({'in_features': 768, 'out_features': 768, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
  (3): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("panagoa/LaBSE-kbd-v0.2")
# Run inference
sentences = [
    "We've never seen Tom this angry before.",
    "Tom'u daha önce asla bu kadar öfkeli görmedik.",
    'Soyez attentive aux voleurs à la tire.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `validation`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value       |
|:--------------------|:------------|
| pearson_cosine      | -0.28       |
| **spearman_cosine** | **-0.3212** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 3,395,988 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                           |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                           |
  | details | <ul><li>min: 5 tokens</li><li>mean: 10.33 tokens</li><li>max: 50 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 13.81 tokens</li><li>max: 46 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.36</li><li>max: 0.98</li></ul> |
* Samples:
  | sentence_0                             | sentence_1                                  | label                           |
  |:---------------------------------------|:--------------------------------------------|:--------------------------------|
  | <code>Почему вас это удивило?</code>   | <code>Сыт ар щIывгъэщIэгъуар?</code>        | <code>0.9298050403594972</code> |
  | <code>Ребёнка кто-нибудь видел?</code> | <code>Quelqu'un a-t-il vu l'enfant ?</code> | <code>0.0</code>                |
  | <code>Marie se couchait.</code>        | <code>Мэри гъуэлъырт.</code>                | <code>0.9330472946166992</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss | validation_spearman_cosine |
|:------:|:-----:|:-------------:|:--------------------------:|
| 0.0005 | 100   | -             | -0.7761                    |
| 0.0009 | 200   | -             | -0.7598                    |
| 0.0014 | 300   | -             | -0.7485                    |
| 0.0019 | 400   | -             | -0.7412                    |
| 0.0024 | 500   | 0.2864        | -0.7354                    |
| 0.0028 | 600   | -             | -0.7307                    |
| 0.0033 | 700   | -             | -0.7191                    |
| 0.0038 | 800   | -             | -0.7206                    |
| 0.0042 | 900   | -             | -0.7197                    |
| 0.0047 | 1000  | 0.0463        | -0.7037                    |
| 0.0052 | 1100  | -             | -0.6866                    |
| 0.0057 | 1200  | -             | -0.6798                    |
| 0.0061 | 1300  | -             | -0.6844                    |
| 0.0066 | 1400  | -             | -0.6716                    |
| 0.0071 | 1500  | 0.0184        | -0.6658                    |
| 0.0075 | 1600  | -             | -0.6620                    |
| 0.0080 | 1700  | -             | -0.6532                    |
| 0.0085 | 1800  | -             | -0.6455                    |
| 0.0090 | 1900  | -             | -0.6452                    |
| 0.0094 | 2000  | 0.011         | -0.6360                    |
| 0.0099 | 2100  | -             | -0.6240                    |
| 0.0104 | 2200  | -             | -0.6220                    |
| 0.0108 | 2300  | -             | -0.6294                    |
| 0.0113 | 2400  | -             | -0.6038                    |
| 0.0118 | 2500  | 0.0092        | -0.6116                    |
| 0.0122 | 2600  | -             | -0.5996                    |
| 0.0127 | 2700  | -             | -0.6120                    |
| 0.0132 | 2800  | -             | -0.5940                    |
| 0.0137 | 2900  | -             | -0.5848                    |
| 0.0141 | 3000  | 0.0071        | -0.5958                    |
| 0.0146 | 3100  | -             | -0.5840                    |
| 0.0151 | 3200  | -             | -0.5944                    |
| 0.0155 | 3300  | -             | -0.5895                    |
| 0.0160 | 3400  | -             | -0.5849                    |
| 0.0165 | 3500  | 0.0056        | -0.5708                    |
| 0.0005 | 100   | -             | -0.5686                    |
| 0.0009 | 200   | -             | -0.5608                    |
| 0.0014 | 300   | -             | -0.5587                    |
| 0.0024 | 500   | 0.0053        | -                          |
| 0.0047 | 1000  | 0.0081        | -0.5882                    |
| 0.0071 | 1500  | 0.0058        | -                          |
| 0.0094 | 2000  | 0.0064        | -0.5127                    |
| 0.0118 | 2500  | 0.004         | -                          |
| 0.0141 | 3000  | 0.0042        | -0.4934                    |
| 0.0165 | 3500  | 0.0048        | -                          |
| 0.0188 | 4000  | 0.0036        | -0.4762                    |
| 0.0212 | 4500  | 0.0051        | -                          |
| 0.0236 | 5000  | 0.0054        | -0.4754                    |
| 0.0259 | 5500  | 0.0054        | -                          |
| 0.0283 | 6000  | 0.0054        | -0.4609                    |
| 0.0306 | 6500  | 0.0044        | -                          |
| 0.0330 | 7000  | 0.0048        | -0.4716                    |
| 0.0353 | 7500  | 0.0061        | -                          |
| 0.0377 | 8000  | 0.0018        | -0.4293                    |
| 0.0400 | 8500  | 0.0047        | -                          |
| 0.0424 | 9000  | 0.0043        | -0.4311                    |
| 0.0448 | 9500  | 0.0034        | -                          |
| 0.0471 | 10000 | 0.0041        | -0.4429                    |
| 0.0495 | 10500 | 0.0028        | -                          |
| 0.0518 | 11000 | 0.0032        | -0.4324                    |
| 0.0542 | 11500 | 0.0025        | -                          |
| 0.0565 | 12000 | 0.0037        | -0.4374                    |
| 0.0589 | 12500 | 0.003         | -                          |
| 0.0612 | 13000 | 0.005         | -0.4522                    |
| 0.0636 | 13500 | 0.0051        | -                          |
| 0.0660 | 14000 | 0.0048        | -0.3994                    |
| 0.0683 | 14500 | 0.0034        | -                          |
| 0.0707 | 15000 | 0.0032        | -0.4148                    |
| 0.0730 | 15500 | 0.0046        | -                          |
| 0.0754 | 16000 | 0.0026        | -0.3848                    |
| 0.0777 | 16500 | 0.0036        | -                          |
| 0.0801 | 17000 | 0.0051        | -0.3845                    |
| 0.0824 | 17500 | 0.0031        | -                          |
| 0.0848 | 18000 | 0.0035        | -0.3500                    |
| 0.0872 | 18500 | 0.0028        | -                          |
| 0.0895 | 19000 | 0.0021        | -0.3634                    |
| 0.0919 | 19500 | 0.0025        | -                          |
| 0.0942 | 20000 | 0.0023        | -0.3428                    |
| 0.0966 | 20500 | 0.0042        | -                          |
| 0.0989 | 21000 | 0.0038        | -0.3432                    |
| 0.1013 | 21500 | 0.005         | -                          |
| 0.1037 | 22000 | 0.0024        | -0.3515                    |
| 0.1060 | 22500 | 0.0029        | -                          |
| 0.1084 | 23000 | 0.0033        | -0.3929                    |
| 0.1107 | 23500 | 0.003         | -                          |
| 0.1131 | 24000 | 0.0029        | -0.3309                    |
| 0.1154 | 24500 | 0.0038        | -                          |
| 0.1178 | 25000 | 0.0028        | -0.3369                    |
| 0.1201 | 25500 | 0.0025        | -                          |
| 0.1225 | 26000 | 0.002         | -0.3257                    |
| 0.1249 | 26500 | 0.0025        | -                          |
| 0.1272 | 27000 | 0.0033        | -0.3659                    |
| 0.1296 | 27500 | 0.0023        | -                          |
| 0.1319 | 28000 | 0.0031        | -0.3208                    |
| 0.1343 | 28500 | 0.0027        | -                          |
| 0.1366 | 29000 | 0.0031        | -0.3298                    |
| 0.1390 | 29500 | 0.0047        | -                          |
| 0.1413 | 30000 | 0.003         | -0.3460                    |
| 0.1437 | 30500 | 0.004         | -                          |
| 0.1461 | 31000 | 0.0027        | -0.3567                    |
| 0.1484 | 31500 | 0.0063        | -                          |
| 0.1508 | 32000 | 0.003         | -0.3382                    |
| 0.1531 | 32500 | 0.0022        | -                          |
| 0.1555 | 33000 | 0.0048        | -0.3475                    |
| 0.1578 | 33500 | 0.0021        | -                          |
| 0.1602 | 34000 | 0.0043        | -0.3323                    |
| 0.1625 | 34500 | 0.0031        | -                          |
| 0.1649 | 35000 | 0.0024        | -0.3207                    |
| 0.1673 | 35500 | 0.0029        | -                          |
| 0.1696 | 36000 | 0.0032        | -0.3004                    |
| 0.1720 | 36500 | 0.0046        | -                          |
| 0.1743 | 37000 | 0.0033        | -0.3085                    |
| 0.1767 | 37500 | 0.002         | -                          |
| 0.1790 | 38000 | 0.0022        | -0.3270                    |
| 0.1814 | 38500 | 0.0036        | -                          |
| 0.1837 | 39000 | 0.0034        | -0.3042                    |
| 0.1861 | 39500 | 0.0034        | -                          |
| 0.1885 | 40000 | 0.0016        | -0.3193                    |
| 0.1908 | 40500 | 0.0026        | -                          |
| 0.1932 | 41000 | 0.0028        | -0.2945                    |
| 0.1955 | 41500 | 0.0031        | -                          |
| 0.1979 | 42000 | 0.0016        | -0.2942                    |
| 0.2002 | 42500 | 0.0021        | -                          |
| 0.2026 | 43000 | 0.003         | -0.2998                    |
| 0.2049 | 43500 | 0.0042        | -                          |
| 0.2073 | 44000 | 0.0023        | -0.3245                    |
| 0.2097 | 44500 | 0.0018        | -                          |
| 0.2120 | 45000 | 0.0021        | -0.3212                    |

</details>

### Framework Versions
- Python: 3.11.11
- Sentence Transformers: 3.4.1
- Transformers: 4.48.3
- PyTorch: 2.5.1+cu124
- Accelerate: 1.3.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->