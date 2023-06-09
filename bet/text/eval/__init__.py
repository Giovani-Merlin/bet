import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm


from bet.text.datasets.retrieval_raw_dataset import (
    RetrievalRawCandidatesDataset,
    RetrievalRawQueriesDataset,
)

from bet.text.eval.stats import compute_statistics
from bet.text.model.model import CandidateEncoder, QueryEncoder

logger = logging.getLogger("bet")


def full_eval(params, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading candidate encoder")
    candidate_encoder = CandidateEncoder.load_model(
        model_path=params["candidate_encoder_weights_path"], device=device
    )
    #
    candidate_encoder = candidate_encoder.to(device)
    logger.info("Loading candidates dataset")
    candidates_dataset = RetrievalRawCandidatesDataset(
        params, samples_to_use=1e6
    ).raw_data_set
    abstracts = [data["abstract"] for data in candidates_dataset]
    # Needs correct index for benchmark
    index_ids = [data["candidate_index"] for data in candidates_dataset]

    #
    # index = None
    index = candidate_encoder.load_index(index_path=params["testing_index_path"])
    if index is None:
        logger.info("Encoding candidates")
        index = candidate_encoder.create_index(
            sentences=abstracts,
            device=device,
            index_ids=index_ids,
            index_configs={"brute_force": False},
            batch_size=params["testing_batch_size"],
        )
        candidate_encoder.save_index(params["testing_index_path"])
        candidates_title = [data["candidate"] for data in candidates_dataset]
        # Useful index_to_title for inference/qualitative analysis
        index_to_title = {
            index: title for index, title in zip(index_ids, candidates_title)
        }
        with open(
            os.path.join(params["testing_index_path"], "index_to_title.json"), "w"
        ) as f:
            json.dump(index_to_title, f)

    # Get the closest candidates for each query
    logger.info("Loading query encoder")
    query_encoder = QueryEncoder.load_model(
        model_path=params["query_encoder_weights_path"], device=device
    )
    query_encoder = query_encoder.to(device)
    logger.info("Loading queries dataset")
    # ! TODO(GM): Maybe do it in batches as the dataset can be huge. To avoid memory problems
    queries_dataset = RetrievalRawQueriesDataset(
        dataset_name=dataset_name, params=params
    ).raw_data_set
    # Get 100k random queries
    queries_dataset = (
        np.random.choice(queries_dataset, 100000, replace=False).tolist()
        if len(queries_dataset) > 100000
        else queries_dataset
    )
    logger.info("Processing queries dataset")
    logger.info("Encoding queries")
    encoded_queries = query_encoder.encode(queries_dataset, device=device)
    # Input encoded queries in the dataset
    for query_ds, encoded_query in zip(queries_dataset, encoded_queries):
        query_ds["encoded_query"] = encoded_query
    # Get the closest candidates for each query
    logger.info("Getting closest candidates")
    # Put tqdm here and do it in batches... Too slow otherwise
    batch_size = 1000
    for i in tqdm(range(0, len(queries_dataset), batch_size)):
        encoded_queries = [
            query["encoded_query"] for query in queries_dataset[i : i + batch_size]
        ]
        candidates, distances = candidate_encoder.search(
            encoded_query=np.array(encoded_queries), top_k=1000, index=index
        )
        # update the queries dataset
        for candidates_query, query_ds, distance in zip(
            candidates, queries_dataset[i : i + batch_size], distances
        ):
            query_ds["closest_indexes"] = candidates_query
            query_ds["distances"] = distance

    # 'candidate_index' is the correct index for each query. 'closest_indexes' are the indexes of the closest candidates
    # Get the recall for each query
    logger.info("Getting recall")
    for query in queries_dataset:
        correct_position = np.where(
            query["closest_indexes"] == query["candidate_index"]
        )[0]
        query["correct_position"] = correct_position[0] if len(correct_position) else -1
    #
    correct_positions = [query["correct_position"] for query in queries_dataset]
    distances = [query["distances"] for query in queries_dataset]
    statistics = compute_statistics(correct_positions, distances)
    # Save statistics
    with open(
        os.path.join(params["output_path"], f"{dataset_name}_statistics.json"), "w"
    ) as f:
        json.dump(statistics, f)

    return statistics
