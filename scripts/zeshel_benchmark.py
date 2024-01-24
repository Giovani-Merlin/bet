"""
    Needs to run examples.zeshel.split_by_worlds.py first
    Not optimized! It will reload the model for each world.
"""

import logging
import os

from bet.bet_parser import BetParser
from bet.text.eval import full_eval
from glob import glob
from collections import defaultdict
import numpy as np
import json

logger = logging.getLogger("bet")
domain_set = {}
domain_set["validation"] = set(
    ["coronation_street", "muppets", "ice_hockey", "elder_scrolls"]
)
domain_set["test"] = set(["forgotten_realms", "lego", "star_trek", "yugioh"])
domain_set["train"] = set(
    [
        "american_football",
        "doctor_who",
        "fallout",
        "final_fantasy",
        "military",
        "pro_wrestling",
        "starwars",
        "world_of_warcraft",
    ]
)
inversed_domain_set = {}
for k, v in domain_set.items():
    inversed_domain_set.update({vv: k for vv in v})


if __name__ == "__main__":
    parser = BetParser(eval=True)
    params = parser.parse_groups()

    worlds = os.listdir(params["data_data_path"])
    data_path = params["data_data_path"]
    world_results = defaultdict(dict)

    output_path = params["output_path"]
    for world in worlds:
        params["data_data_path"] = os.path.join(data_path, world)
        params["output_path"] = os.path.join(output_path, world)
        params["testing_index_path"] = os.path.join(output_path, world, "index")
        world_results[world] = full_eval(params, "queries")
    # Group by dataset per model
    dataset_results = defaultdict(dict)

    stats_world = defaultdict(list)
    for world, stats in world_results.items():
        stats_world[inversed_domain_set[world]].append(stats)
    # For each recall, get the mean and std
    dataset_recall_results = defaultdict(dict)

    for dataset, stats in stats_world.items():
        dataset_stats = defaultdict(list)
        for stats_dict in stats:
            for recall, value in stats_dict["recall"].items():
                dataset_stats[recall].append(value)
        dataset_recall_results[dataset] = {
            k: {"mean": np.mean(v), "std": np.std(v)} for k, v in dataset_stats.items()
        }
    # Save results of each model

    with open(os.path.join(output_path, "results_zeshel_macro.json"), "w") as f:
        json.dump(dataset_recall_results, f, indent=2)

    print("STOP")
