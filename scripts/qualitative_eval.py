"""
Quickly script to get the top candidates using the same example as the BLINK repo.

"""
import logging
import os

import numpy as np
import torch

from bet.bet_parser import BetParser
from bet.text.model.model import QueryEncoder
import json

logger = logging.getLogger("bet")


if __name__ == "__main__":
    parser = BetParser(eval=True)
    params = parser.parse_groups()

    output_path = params["output_path"]
    os.makedirs(params["output_path"], exist_ok=True)
    # Expected format:
    # 'mention', 'query_left', 'query_right'
    blink_example = "Bert and Ernie are two Muppets who appear together in numerous skits on the popular children's television show of the United States, Sesame Street."
    entities = ["Bert", "Ernie", "Muppets", "United States", "Sesame Street"]
    as_blink_format = []
    for entity in entities:
        split = blink_example.split(entity)
        context_left = split[0]
        context_right = split[1]
        as_blink_format.append(
            {
                "link": entity,
                "query_left": context_left,
                "query_right": context_right,
            }
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # No need for candidate encoder - just the index
    query_encoder = QueryEncoder.load_model(
        model_path=params["query_encoder_weights_path"]
    )

    index = query_encoder.load_index(params["testing_index_path"])
    with open(
        os.path.join(params["testing_index_path"], "index_to_title.json"), "r"
    ) as f:
        index_to_title = json.load(f)
        # Fix keys from str to int
        index_to_title = {int(k): v for k, v in index_to_title.items()}
    index.index_to_title = np.vectorize(index_to_title.get)
    query_encoder = query_encoder.to(device)
    candidates, distances = query_encoder.search(
        query=as_blink_format, top_k=5, index=index, device=device, as_titles=True
    )

    for candidate in candidates:
        print(candidate)
