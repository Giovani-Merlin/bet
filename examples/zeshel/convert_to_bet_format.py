import argparse
import json

import os


def read_dataset(input_path):
    all_datasets = {}
    for type_ in ["test.jsonl", "train.jsonl", "valid.jsonl"]:
        file_name = os.path.join(input_path, type_)
        with open(file_name, mode="r", encoding="utf-8") as file:
            dataset_data = []
            for n, line in enumerate(file):
                dataset_data.append(json.loads(line.strip()))
        all_datasets[type_] = dataset_data
    return all_datasets


def map_zeshel_to_wbdsm(dataset, output_path):
    """
    Map zeshel to bdsm format.
    For that we need to first get all the candidates for each query - to create unique candidate ids.

    """
    candidates_pool = []
    candidates_title = []
    n = 0
    for type_ in ["test.jsonl", "train.jsonl", "valid.jsonl"]:
        for sample in dataset[type_]:
            candidate_title = sample["label_title"]
            if candidate_title not in candidates_pool:
                item = {}
                item["abstract"] = sample["label"]
                item["candidate_title"] = sample["label_title"]
                item["candidate_index"] = n
                item["world"] = sample["world"]
                n += 1
                candidates_pool.append(item)
                candidates_title.append(candidate_title)
    print(f"Number of candidates: {len(candidates_pool)}")
    # Now we have all the candidates, we can create a mapping from candidate title to candidate id
    # And perform the other mappings
    k = 0
    for type_ in ["test.jsonl", "train.jsonl", "valid.jsonl"]:
        if type_ == "valid.jsonl":
            output_file_name = os.path.join(output_path, "validation.jsonl")
        else:
            output_file_name = os.path.join(output_path, type_)

        with open(output_file_name, mode="w", encoding="utf-8") as file:
            for sample in dataset[type_]:
                item = {}
                item["query_left"] = sample["context_left"]
                item["query_right"] = sample["context_right"]
                item["mention"] = sample["mention"]
                item["candidate_index"] = candidates_title.index(sample["label_title"])
                item["query_index"] = k
                item["world"] = sample["world"]
                k += 1
                file.write("{}\n".format(json.dumps(item)))
        print(f"Finished writing {type_}")
    # Saves candidate pool
    with open(
        os.path.join(output_path, "candidates.jsonl"), mode="w", encoding="utf-8"
    ) as file:
        for candidate in candidates_pool:
            file.write("{}\n".format(json.dumps(candidate)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path",
        default="data/zeshel/wbdsm_format",
        type=str,
    )
    parser.add_argument(
        "--input_path",
        default="data/zeshel/blink_format",
        type=str,
    )
    params = parser.parse_args()
    os.makedirs(params.output_path, exist_ok=True)
    dataset = read_dataset(params.input_path)
    map_zeshel_to_wbdsm(dataset, params.output_path)
    print("Done")
