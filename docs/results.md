# Zeshel

First, get Zeshel dataset following this [small tutorial](examples/zeshel/README.md).

## Pre training

For pre-training, first we need to create the full wikipedia pre-training dataset. For that, please follow [WBDSM entity linking dataset generation](https://github.com/Giovani-Merlin/wbdsm/blob/main/docs/entity_linking_dataset_generation.md). Using the arguments:

```json
{ "language": "en", "max_chars": 2048, "candidates_size": 2000000, "train_size": 100000, "test_size": 1000, "validation_size": 1000, "candidate_text_surfaces": 1, "candidate_surface_appearance": 10, "max_rank": 100000,  "query_max_chars": 1024}
'''
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
'''

## Fine-tuning

Having the pre-trained model, we can fine-tune it on the zeshel dataset using the following arguments:

'''bash
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
'''
Finally, we can evaluate it in the same way as BLINK - macro results per world.

'''bash
"--data_data_path","data/zeshel/wbdsm_format/split_by_worlds",
"--query_encoder_weights_path","models/zeshel/bert-medium",
"--candidate_encoder_weights_path","models/zeshel/bert-medium",
"--output_path","models/zeshel/bert-medium",
'''

## Results:

# Note

The qualitative results in the other side, are not so great. The model seems great for sentence similarity tasks (with the option to "focus" on some parts by using the entity envelope) but not so great for entity linking. The main problem is that the model is because of the strategy of the dataset creation - we have mostly different context with just one entity - and using one-shot it can't rely in just the entity surface' , therefore, the models learn to map the context with just a small focus on the entity. This is a known problem that must be fixed at the dataset time creation - we just need to sample different candidates relying in some score to give preference to the same Wikipedia page and section.
