"""
# * Default class for data processing (using from pytorch multiprocessing)
"""
import json
import logging
import os
from typing import Dict

import numpy as np
import tables
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from bet.text.datasets.constants import ENT_END_TAG, ENT_START_TAG, ENT_TITLE_TAG
from bet.text.datasets.retrieval_dataset import RetrievalDataset
from bet.text.datasets.utils import (
    get_candidate_representation,
    get_dataloader,
    get_dataloader_iter,
    get_hdf5_dataset_dict,
    get_query_representation,
    group_processed_candidates,
    group_processed_queries,
)

logger = logging.getLogger("bet")


class RetrievalRawQueriesDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        params: Dict[str, str],
    ):
        self.dataset_name = dataset_name
        self.params = params
        self.raw_data_set = self.read_dataset()
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            params["query_encoder_model"]
        )
        self.query_tokenizer.add_special_tokens(
            {"additional_special_tokens": [ENT_START_TAG, ENT_END_TAG]}
        )

    def __len__(self):
        return len(self.raw_data_set)

    def __getitem__(self, index):
        processed_data = self.process_data(self.raw_data_set[index])
        return processed_data

    def read_dataset(self):
        """
        Read dataset from jsonl file. If debug, only use 200 first samples.
        """
        samples = []
        file_name = f"{self.dataset_name}.jsonl"
        txt_file_path = os.path.join(self.params["data_data_path"], file_name)

        with open(txt_file_path, mode="r", encoding="utf-8") as file:
            for n, line in enumerate(file):
                # if debug, only use 200 first samples
                if self.params["training_debug"] and n >= 100:
                    break
                samples.append(json.loads(line.strip()))
        return samples

    def process_data(self, data: Dict[str, str]):
        """
        Recover query and candidate representation from raw data.
        For candidate representation, we can use the candidate title or not and just truncate the candidate description.
        For query representation, we center the mention in the query and truncate the query.
        """
        query_representation = self.get_query_representation(data)

        candidate_id = data["candidate_index"]
        query_id = data["query_index"]
        record = {
            "query": query_representation,
            "candidate_index": candidate_id,
            "query_index": query_id,
        }

        return record

    def get_query_representation(self, sample: Dict[str, str]):
        return get_query_representation(
            sample,
            self.query_tokenizer,
            self.params["data_query_max_length"],
            "link",
            "query",
        )


class RetrievalRawCandidatesDataset(Dataset):
    """
    Utility class for processing raw dataset with multiprocessing. Also creates tokenizers with special tokens.
    """

    def __init__(
        self,
        params: Dict[str, str],
        samples_to_use: int = None,
    ):
        self.params = params
        self.raw_data_set = self.read_dataset(samples_to_use)
        # ! Create tokenizers with special tokens
        self.candidate_tokenizer = AutoTokenizer.from_pretrained(
            params["candidate_encoder_model"]
        )
        self.candidate_tokenizer.add_special_tokens(
            {"additional_special_tokens": [ENT_TITLE_TAG]}
        )

    def __len__(self):
        return len(self.raw_data_set)

    def __getitem__(self, index):
        processed_data = self.process_data(self.raw_data_set[index])
        return processed_data

    def read_dataset(self, samples_to_use: int = None):
        """
        Read dataset from jsonl file. If debug, filter to only use the candidates from the queries set.
        """
        samples = []
        file_name = "candidates.jsonl"
        txt_file_path = os.path.join(self.params["data_data_path"], file_name)

        with open(txt_file_path, mode="r", encoding="utf-8") as file:
            for n, line in enumerate(file):
                samples.append(json.loads(line.strip()))
                if samples_to_use and n >= samples_to_use:
                    break
        return samples

    def process_data(self, data: Dict[str, str]):
        """
        Recover query and candidate representation from raw data.
        For candidate representation, we can use the candidate title or not and just truncate the candidate description.
        For query representation, we center the mention in the query and truncate the query.
        """

        candidate_representation = self.get_candidate_representation(data)
        candidate_id = data["candidate_index"]
        record = {
            "candidate": candidate_representation,
            "candidate_index": candidate_id,
        }

        return record

    def get_candidate_representation(
        self,
        sample: Dict[str, str],
    ):
        return get_candidate_representation(
            sample["abstract"],
            self.candidate_tokenizer,
            self.params["data_candidate_max_length"],
            self.params.get("data_candidate_title"),
        )


####
###
####
def create_hdf5_dataset_candidates(iter_: str, file_dicts: str, dataset_size: int):
    """
    Create h5py file to store candidates data.
    We could just store in an ordered way - as we have candidate line = candidate index. For safety, we store it using the candidate index.

    """
    first_batch = True
    # ! Process and store data in h5py files
    for batch in iter_:
        samples = batch
        candidates_index = samples.pop("candidate_index")
        candidate = samples.pop("candidate")
        # Create datasets
        if first_batch:
            for key, value in candidate.items():
                file_dicts["candidate"].create_dataset(
                    key,
                    (dataset_size, len(value[0]))
                    if type(value[0]) == np.ndarray
                    else (dataset_size,),
                    dtype=value[0].dtype
                    if type(value[0]) == np.ndarray
                    else value[0].__class__,
                )
            first_batch = False

        # Store data in the datasets
        for key, value in candidate.items():
            file_dicts["candidate"][key][candidates_index] = value


def create_hdf5_dataset_queries(iter_: str, file_dicts: str, dataset_size: int):
    first_batch = True
    index = 0
    # ! Process and store data in h5py files
    for batch in iter_:
        samples = batch
        batch_size = len(samples["auxiliar"]["candidate_index"])
        final_index = index + batch_size
        # Create datasets
        if first_batch:
            for dataset, values in samples.items():
                # Specific datasets
                for key, value in values.items():
                    file_dicts[dataset].create_dataset(
                        key,
                        (dataset_size, len(value[0]))
                        if type(value[0]) == np.ndarray
                        else (dataset_size,),
                        dtype=value[0].dtype
                        if type(value[0]) == np.ndarray
                        else value[0].__class__,
                    )
            first_batch = False
        # Store data in the datasets

        for dataset, values in samples.items():
            for key, value in values.items():
                file_dicts[dataset][key][index:final_index] = value
        index = final_index


# * Utilities functions for data processing pipeline
def process_raw_dataset(dataset_type: str, params: Dict[str, str]):
    """
    Process raw dataset using RetrievalRawDataset helper class
    :return:
    list of tokenized query and candidates.
    """

    queries_dataset = RetrievalRawQueriesDataset(dataset_type, params)
    queries_iter = get_dataloader_iter(
        queries_dataset, params["data_workers"], group_processed_queries
    )
    # Create h5py file to store query and candidates data
    file_dicts = get_hdf5_dataset_dict(params, dataset_type, "w")
    create_hdf5_dataset_queries(
        queries_iter, file_dicts, len(queries_dataset.raw_data_set)
    )
    # Process only one time the candidates pool.
    if dataset_type == "train":
        # Process candidates dataset
        candidates_dataset = RetrievalRawCandidatesDataset(params)
        candidates_iter = get_dataloader_iter(
            candidates_dataset, params["data_workers"], group_processed_candidates
        )
        create_hdf5_dataset_candidates(
            candidates_iter, file_dicts, len(candidates_dataset.raw_data_set)
        )

    # Close h5py files
    for file_dict in file_dicts.values():
        file_dict.close()


###
### lightning data module for training
###


# using LightningDataModule
class RetrievalDataModule(LightningDataModule):
    """RetrievalDataModule class for loading data"""

    def __init__(self, training_params: Dict[str, str]):
        """Initialize the RetrievalDataModule class with training parameters"""
        super().__init__()
        self.batch_size = training_params["training_batch_size"]
        self.save_hyperparameters()
        self.params = training_params
        self.seed = self.params["training_seed"]
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)

    def setup(self, stage):
        """Setup the datasets for training, validation and testing"""
        for dataset_type in ["train", "validation", "test"]:
            process_dataset_if_needed(self.params, dataset_type)

    def train_dataloader(self):
        """Return a dataloader for training"""
        dataset_dict = get_hdf5_dataset_dict(self.params, "train", "r")
        retrieval_dataset = RetrievalDataset(**dataset_dict)
        data_loader = get_dataloader(
            retrieval_dataset,
            self.params["data_workers"],
            self.params["training_shuffle"],
            self.batch_size,
        )

        return data_loader

    def val_dataloader(self):
        dataset_dict = get_hdf5_dataset_dict(self.params, "validation", "r")
        retrieval_dataset = RetrievalDataset(**dataset_dict)
        data_loader = get_dataloader(
            retrieval_dataset,
            self.params["data_workers"],
            False,
            self.params["testing_batch_size"],
        )
        return data_loader

    def test_dataloader(self):
        dataset_dict = get_hdf5_dataset_dict(self.params, "test", "r")
        retrieval_dataset = RetrievalDataset(**dataset_dict)
        data_loader = get_dataloader(
            retrieval_dataset,
            self.params["data_workers"],
            False,
            self.params["testing_batch_size"],
        )
        return data_loader

    def teardown(self, stage):
        tables.file._open_files.close_all()


def process_dataset_if_needed(params: Dict[str, str], dataset_type: str):
    """
    Process dataset if needed. If cache_path is not none, then we try to load the dataset from cache.
    If it fails, then we process the dataset and store it in cache.
    """
    dataset_dict = None
    if params["data_cache_path"]:
        try:
            dataset_dict = get_hdf5_dataset_dict(params, dataset_type, "r")
        except Exception as e:
            print(f"Error loading dataset from cache: {e}")
            try:
                process_raw_dataset(dataset_type, params)
            except Exception as e:
                print(f"Error processing dataset: {e}")
                raise e
    else:
        process_raw_dataset(dataset_type, params)
    # If dataset_dict is not None, then we have loaded the dataset from cache successfully
    if dataset_dict is not None:
        for file_dict in dataset_dict.values():
            file_dict.close()
