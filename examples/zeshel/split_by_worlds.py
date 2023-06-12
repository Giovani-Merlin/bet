"""
    Pre-processing step to perform full evaluation on Zeshel dataset.
    As we consider that we have just one world, we need to split the dataset by world and then perform the evaluation.
"""
import argparse
import json

import os

from collections import defaultdict


def read_datasets(input_path):
    all_datasets = {}
    for type_ in ["test.jsonl", "train.jsonl", "validation.jsonl"]:
        file_name = os.path.join(input_path, type_)
        with open(file_name, mode="r", encoding="utf-8") as file:
            dataset_data = []
            for n, line in enumerate(file):
                dataset_data.append(json.loads(line.strip()))
        all_datasets[type_] = dataset_data

    with open(
        os.path.join(input_path, "candidates.jsonl"), mode="r", encoding="utf-8"
    ) as file:
        dataset_data = []
        for n, line in enumerate(file):
            dataset_data.append(json.loads(line.strip()))
    all_datasets["candidates"] = dataset_data
    return all_datasets


def split_by_worlds(datasets):
    """
    Split the dataset by worlds.
    """
    queries_by_worlds = defaultdict(list)
    for type_ in ["test.jsonl", "train.jsonl", "validation.jsonl"]:
        for sample in datasets[type_]:
            world = sample["world"]
            queries_by_worlds[world].append(sample)
    # now split the candidates
    candidates_by_worlds = defaultdict(list)
    for sample in datasets["candidates"]:
        world = sample["world"]
        candidates_by_worlds[world].append(sample)

    return queries_by_worlds, candidates_by_worlds


def save_per_world(queries, candidates, output_path):
    """
    Save the queries and candidates in a per world format.
    """
    for world in queries.keys():
        world_queries = queries[world]
        world_candidates = candidates[world]
        os.makedirs(os.path.join(output_path, world), exist_ok=True)
        with open(
            os.path.join(output_path, world, "queries.jsonl"),
            mode="w",
            encoding="utf-8",
        ) as file:
            for query in world_queries:
                file.write(json.dumps(query) + "\n")
        with open(
            os.path.join(output_path, world, "candidates.jsonl"),
            mode="w",
            encoding="utf-8",
        ) as file:
            for candidate in world_candidates:
                file.write(json.dumps(candidate) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path",
        default="data/zeshel/wbdsm_format/split_by_worlds",
        type=str,
    )
    parser.add_argument(
        "--input_path",
        default="data/zeshel/wbdsm_format",
        type=str,
    )
    params = parser.parse_args()
    os.makedirs(params.output_path, exist_ok=True)
    datasets = read_datasets(params.input_path)
    queries, candidates = split_by_worlds(datasets)
    save_per_world(queries, candidates, params.output_path)

    print("Done")
