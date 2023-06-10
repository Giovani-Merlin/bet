"""
Should use the already parsed (from training) hdf5 dataset withe the tokenized candidates. 
To make it easy to change dataset evaluation will use RetrievalRawCandidatesDataset directly (and to create the code for inference at the same time)

Needs to remove break on candidates reading for debug

"""
import logging
import os


from collections import defaultdict
from glob import glob
import json
import pandas as pd

logger = logging.getLogger("bet")


if __name__ == "__main__":
    # models_zeshel_path = "models/zeshel"
    models_zeshel_path = "models/en"

    all_models = glob(os.path.join(models_zeshel_path, "*"))
    all_results = {}
    stats_file = "results_zeshel_macro.json"
    stats_file = "eval/test_statistics.json"
    # training_file = "training_results.json"
    for model in all_models:
        # try:
        #     with open(os.path.join(model, training_file)) as f:
        #         stats = json.load(f)
        # except FileNotFoundError:
        #     logger.warning(f"File {training_file} not found for {model}")
        stats = {}
        try:
            with open(os.path.join(model, stats_file)) as f:
                results = json.load(f)
        except FileNotFoundError:
            logger.warning(f"File {stats_file} not found for {model}")
            results = {}
        all_results[os.path.basename(model)] = {**stats, **results}
    # Invert
    df_results = pd.DataFrame.from_dict(all_results, orient="index")
    # Values
    dataset_type = "test"
    dataset_type = "recall"
    dataset_type_values = pd.concat(
        df_results[dataset_type].apply(lambda x: pd.DataFrame(x)).values
    )
    # Recover the model name - terrible parsing but got it...
    dataset_type_values = dataset_type_values.drop("std")
    dataset_type_values.index = df_results[dataset_type].index
    print(dataset_type_values)
