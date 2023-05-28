import logging
import os
from collections import defaultdict
from typing import Dict, List, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer

from bet.text.datasets.constants import ENT_END_TAG, ENT_START_TAG, ENT_TITLE_TAG
from bet.utils import get_iterator, select_field

logger = logging.getLogger("bet")


def get_hdf5_dataset_dict(params: Dict[str, str], dataset_type: str, mode: str):
    """
    :return:
    data_loaders: Dict[str, DataLoader]
    """
    if mode == "w":
        # Create output path
        os.makedirs(params["data_cache_path"], exist_ok=True)
    query_file = h5py.File(
        os.path.join(params["data_cache_path"], f"{dataset_type}_query.h5"), mode
    )
    # ! TODO: Refactor it, this function should just get and use a different one to create it (no "mode")
    # candidates is shared between train, validation and test but just train preprocessing creates it.
    if dataset_type == "train":
        candidate_file = h5py.File(
            os.path.join(params["data_cache_path"], "candidate.h5"), mode
        )
    else:
        candidate_file = h5py.File(
            os.path.join(params["data_cache_path"], "candidate.h5"), "r"
        )
    auxiliar_file = h5py.File(
        os.path.join(params["data_cache_path"], f"{dataset_type}_auxiliar.h5"), mode
    )
    file_dicts = {
        "query": query_file,
        "candidate": candidate_file,
        "auxiliar": auxiliar_file,
    }
    return file_dicts


#
def get_candidate_representation(
    candidate_desc: Union[List[str], str],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    candidate_title: str = None,
    padding="max_length",
    return_tensors="np",
):
    """
    Recover candidate representation from raw data using the tokenizer.
    Also add the candidate title if specified.

    """
    candidate_desc = (
        candidate_desc if isinstance(candidate_desc, list) else [candidate_desc]
    )
    if candidate_title is not None:
        candidate_desc = [
            candidate_title + ENT_TITLE_TAG + cand for cand in candidate_desc
        ]

    cand_inputs = tokenizer.batch_encode_plus(
        candidate_desc,
        max_length=max_seq_length,
        truncation=True,
        padding=padding,
        return_tensors=return_tensors,
    )

    return cand_inputs


def get_query_representation(
    query: Union[Dict[str, str], List[Dict[str, str]]],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    mention_key: str = "mention",
    query_key: str = "query",
    padding="max_length",
    return_tensors="np",
):
    """
    Recover context representation from raw data using the tokenizer.
    It centers the mention in the context and truncate the context.

    TODO: Tokenize all then do the processing to make the tokenization in parallel.
    """
    query = query if isinstance(query, list) else [query]
    query_inputs = []
    for query_dict in query:
        mention_tokens = []
        # Put special mention tokens around mention
        mention_tokens = tokenizer.tokenize(query_dict[mention_key])
        mention_tokens = [ENT_START_TAG] + mention_tokens + [ENT_END_TAG]

        query_left = query_dict[query_key + "_left"]
        query_right = query_dict[query_key + "_right"]
        query_left = tokenizer.tokenize(query_left)
        query_right = tokenizer.tokenize(query_right)

        # Get the maximum length of the context tokens
        left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
        right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
        left_add = len(query_left)
        right_add = len(query_right)
        if left_add <= left_quota:
            if right_add > right_quota:
                right_quota += left_quota - left_add
        else:
            if right_add <= right_quota:
                left_quota += right_quota - right_add

        query_tokens = (
            query_left[-left_quota:] + mention_tokens + query_right[:right_quota]
        )
        final_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(query_tokens))
        query_inputs.append(final_text)
    #
    query_inputs = tokenizer.batch_encode_plus(
        query_inputs,
        max_length=max_seq_length,
        truncation=True,
        padding=padding,
        return_tensors=return_tensors,
    )

    return query_inputs


def group_processed_queries(x):
    """
    Group results from batch_encode_plus and group auxiliar inputs

    Could do almost the same way as retrieval_collate but auxiliar needs to be handled differently
    """
    grouped_data = defaultdict(list)
    # Group all keys, values as a dict
    for row in x:
        for k, v in row.items():
            grouped_data[k].append(v)
    # Group all arguments from batch_encode_plus
    query_data = defaultdict(list)
    for query in grouped_data["query"]:
        for k, v in query.items():
            # Squeeze first dimension from value as it contains the batch size
            v = np.squeeze(v, axis=0)
            query_data[k].append(v)
    grouped_data["query"] = query_data

    data_return = {
        "query": grouped_data.pop("query"),
        "auxiliar": {key: np.array(value) for key, value in grouped_data.items()},
    }
    return data_return


def group_processed_candidates(x):
    """
    Group results from batch_encode_plus and group auxiliar inputs

    Could do almost the same way as retrieval_collate but auxiliar needs to be handled differently
    """
    grouped_data = defaultdict(list)
    # Group all keys, values as a dict
    for row in x:
        for k, v in row.items():
            grouped_data[k].append(v)
    # Group all arguments from batch_encode_plus
    candidate_data = defaultdict(list)
    for candidate in grouped_data["candidate"]:
        for k, v in candidate.items():
            # Squeeze first dimension from value as it contains the batch size
            v = np.squeeze(v, axis=0)
            candidate_data[k].append(v)
    grouped_data["candidate"] = candidate_data
    return grouped_data


def get_dataloader(dataset: Dataset, data_workers: int, shuffle: bool, batch_size: int):
    """
    Helper function to get a dataloader for a dataset. Uses the retrieval_collate function and generator g to keep the same shuffling order between each epoch.
    """
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=retrieval_collate,
        drop_last=True,
        num_workers=data_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )


def retrieval_collate(batch):
    """
    Group outputs by query and candidate
    """
    (query_input, entity_input, auxiliar) = zip(*batch)
    # Group per value
    query_fields = query_input[0].keys()
    query_input = {
        field: torch.tensor(np.vstack(select_field(query_input, field)))
        for field in query_fields
    }
    entity_fields = entity_input[0].keys()
    entity_input = {
        field: torch.tensor(np.vstack(select_field(entity_input, field)))
        for field in entity_fields
    }
    auxiliar_fields = auxiliar[0].keys()
    auxiliar = {
        field: np.vstack(select_field(auxiliar, field)) for field in auxiliar_fields
    }

    return query_input, entity_input, auxiliar


def get_dataloader_iter(dataset, data_workers, collate_fn=None):
    """
    Helper function to get a dataloader for a dataset. Uses the retrieval_collate function. Used to preprocess queries and candidates.
    """
    sampler = SequentialSampler(dataset.raw_data_set)
    process_batch_size = 256
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=process_batch_size,
        num_workers=data_workers,
        collate_fn=collate_fn,  # lambda x is not pickable
    )

    iter_ = get_iterator(
        dataloader, f"Processing data", n_samples=len(dataset.raw_data_set)
    )
    return iter_
