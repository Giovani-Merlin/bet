# Zeshel

First, get Zeshel dataset following this [small tutorial](examples/zeshel/README.md).

## Pre training

For pre-training, first we need to create the full wikipedia pre-training dataset. For that, please follow [WBDSM entity linking dataset generation](https://github.com/Giovani-Merlin/wbdsm/blob/main/docs/entity_linking_dataset_generation.md). Using the arguments:

```json
{ "language": "en", "max_chars": 2048, "candidates_size": 2000000, "train_size": 100000, "test_size": 1000, "validation_size": 1000, "candidate_text_surfaces": 1, "candidate_surface_appearance": 10, "max_rank": 100000,  "query_max_chars": 1024}
```

The dumped used was 'enwiki-20230401-pages-articles-multistream.xml.bz2'

The generated dataset has approximately 900.000 rows for train, and 5.000 for train and test. The dataset is already on BET format, so we can just use it to train the model.
The idea of this dataset was to force the most "one-shot" possible by using 'candidate_surface_appearance': 10 and 'candidate_text_surfaces': 1. That is, we have at most 10 times each candidate and just one time each candidate surface.

After having the dataset, we can pre-train the model using the following arguments with script 'scripts/train.py':

```bash
    "--data_data_path","data/en/bert-medium",
    "--output_path","models/en/bert-medium", 
    "--data_cache_path","models/en/bert-medium",
    "--training_debug","False",
    "--training_val_check_interval","0.5",
    "--training_max_epochs","10",
    "--training_min_epochs","5",
    "--training_batch_size","96",
    "--training_auto_batch_size","False",
    "--candidate_encoder_model","prajjwal1/bert-medium",
    "--query_encoder_model","prajjwal1/bert-medium",
    "--training_random_negatives_loss_scaler","90.00",
    "--testing_eval_recall","1"
```

## Fine-tuning

Having the pre-trained model, we can fine-tune it on the zeshel dataset using the following arguments:

```bash
"--data_data_path","data/zeshel/wbdsm_format",
"--query_encoder_weights_path","models/en/bert-medium",
"--candidate_encoder_weights_path","models/en/bert-medium",
"--data_cache_path","data/zeshel/wbdsm_format/bert_processed",
"--output_path","models/zeshel/bert-medium",
"--data_use_cache","True",
"--training_val_check_interval","0.5",
"--training_max_epochs","2",
"--training_batch_size","96",
"--training_auto_batch_size","False",
"--training_random_negatives_loss_scaler","90.00"
```

Finally, we can evaluate it in the same way as BLINK - macro results per world with the script on `scripts/zeshel_benchmark.py`

```bash
"--data_data_path","data/zeshel/wbdsm_format/split_by_worlds",
"--query_encoder_weights_path","models/zeshel/bert-medium",
"--candidate_encoder_weights_path","models/zeshel/bert-medium",
"--output_path","models/zeshel/bert-medium",
```

## Results

|       | Train R@1 | Validation R@1 | Test R@1 | Train R@64 | Validation R@64 | Test R@1 |
| ----- | --------- | -------------- | -------- | ---------- | --------------- | -------- |
| Blink | x         | x              | 63.03    | 93.12      | 91.44           | 82.06    |
| BET   | 78        | 68.5           | 68       | 98.12      | 94.2            | 90       |

All results from BLINK where colected from their paper. Results from BET where made by macro-averaging the results for all world results in the given dataset type (as BLINK).

We can see huge improvement in all the datasets, this improvement probably is due to the methodology choosen for creating the pre-training dataset, that is, it was pre-trained with a hard dataset, being force to learn the context of the data and not memorizing the candidate's surfaces.

## Ablation for candidates pool size

A quick study was made to see how the candidates pool size affects the results. These results were obtained using the german model.

Considering the train set, we see how the recall is affected by the candidates pool size. It starts with 77k candidates as the train set contained 77k candidates.

with 77k candidates
1: 0.91, 2: 0.94, 8:0.98 ,16: 0.99
with 150k
1: 0.87 , 2: 0.91 , 8: 0.96 , 16: 0.98
with 300k:
1: 0.84, 2:0.88 , 8: 0.95 . 16: 0.97

As for the test set (starting with 5k candidates)

5k
1:0.96 , 2: 0.97 , 8: 0.98 , 16: 0.99
10k:
1: 0.94, 2:0.95 . 8: 0.97 , 16: 0.98
20k:
1: 0.91, 2: 0.93, 8: 0.96, 16: 0.97
50k:
1: 0.88, 2: 0.90, 8: 0.94, 16: 0.96
150k:
1: 0.83, 2: 0.86, 8: 0.91, 16: 0.94
300k:
1: 0.80, 2: 0.84, 8: 0.90, 16: 0.92

We observe a consistent parallel trend in the progression of both the test and train sets. However, the recall of the train set is consistently higher.

It is essential to carefully choose the size of the candidate pool to ensure a high recall rate. Moreover, this observation highlights the excellent performance of the model in the test set, especially when limiting the candidate pool to non-seen candidates. Therefore, using a pre-trained model for entity linking in new domains with a restricted candidate pool is an effective approach.
