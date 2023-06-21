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

|   | Train R@1  | Validation R@1  | Test R@1  | Train R@64  | Validation R@64  | Test R@1  |
|--- |--- |--- |--- |--- |--- |--- |
| Blink  | x  | x  | 63.03  | 93.12  | 91.44  | 82.06  |
| BET  | 78  | 68.5  | 68  | 98.12  | 94.2  | 90  |

All results from BLINK where colected from their paper. Results from BET where made by macro-averaging the results for all world results in the given dataset type (as BLINK).

We can see huge improvement in all the datasets, this improvement probably is due to the methodology choosen for creating the pre-training dataset, that is, it was pre-trained with a hard dataset, being force to learn the context of the data and not memorizing the candidate's surfaces.
