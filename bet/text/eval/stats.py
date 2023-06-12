"""
    Helper functions to perform benchmarking on the datasets
"""
from collections import defaultdict
from typing import List

import numpy as np


def get_quantile_data(data, max_cand):
    bins = [1, 2, 3, 6, 11, 21, 51, 101, 501, max_cand]
    bins_legend = [
        "0",
        "1",
        "2",
        "3-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        "101-500",
        f"501-{max_cand}",
    ]
    quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    quantiles_legend = ["1%", "10%", "25%", "50%", "75%", "90%", "99%"]
    last = data[-1]
    summary = {}
    for bin_end, legend in zip(bins, bins_legend):
        accumulated_distances = []
        entries = 0
        while data and data[0][0] < bin_end:
            entries += len(data[0][1])
            accumulated_distances.extend(data.pop(0)[1])
        if accumulated_distances:
            quantiles_data = np.quantile(accumulated_distances, quantiles).tolist()
        else:
            quantiles_data = "No entry"
        summary[legend] = {
            "quantiles": quantiles_data,
            "avg": str(np.mean(accumulated_distances)),
            "quantiles_legend": quantiles_legend,
            "total": entries,
        }

    # Get (if exist) out entities
    if last[0] > max_cand:
        quantiles_data = np.quantile(last[1], quantiles).tolist()
        summary[f">{max_cand}"] = {
            "quantiles": quantiles_data,
            "avg": str(np.mean(last[1])),
            "quantiles_legend": quantiles_legend,
            "total": last[1],
        }
    return summary


def compute_statistics(positions: List[int], distances: List[List[float]]):
    # Needs to filter hard-coded recalls if the number of candidates is smaller
    max_cand = max([len(distance) for distance in distances])
    recalls = [1, 2, 4, 8, 16, 32, 64, 100, 128, 300, 1000]
    recalls = [r for r in recalls if r < max_cand]
    #
    positions = np.array(positions)
    # Change -1 (not in the results) to max_cand + 1
    positions[positions == -1] = max_cand + 1
    recall_results = defaultdict(int)
    # could be optimized but it's fast anyway
    # if make np.array(positions) we have overflow in the memory
    for recall in recalls:
        for position in positions:
            if position <= recall:
                recall_results[recall] += 1
        recall_results[recall] /= len(positions)
    # Distance of the first entity
    distance_results = defaultdict(list)
    # Distance of the correct entity to the next one
    diff_distance_results = defaultdict(list)
    # If correct, distance to the correct entity
    first_distance_results = defaultdict(list)
    for distance, position in zip(distances, positions):
        first_distance_results[position].append(float(distance[0]))
        if position < max_cand:
            distance_results[position].append(distance[position])
            if position < max_cand - 1:
                diff_distance_results[position].append(
                    distance[position] - distance[position + 1]
                )
        else:
            # if the position is greater than the max_cand, we use the last distance
            distance_results[max_cand].append(distance[-1])
            # And we get the distance from last to first
            diff_distance_results[max_cand].append(distance[-1] - distance[0])
    distance_results = sorted(distance_results.items(), key=lambda x: x[0])
    first_distance_results = sorted(first_distance_results.items(), key=lambda x: x[0])
    diff_distance_results = sorted(diff_distance_results.items(), key=lambda x: x[0])
    distance_stats = get_quantile_data(distance_results, max_cand)
    diff_stats = get_quantile_data(diff_distance_results, max_cand)
    first_distance_stats = get_quantile_data(first_distance_results, max_cand)
    # Transform as dict the defaultdict
    recall_results = {k: v for k, v in recall_results.items()}
    distance_stats = {k: v for k, v in distance_stats.items()}
    diff_stats = {k: v for k, v in diff_stats.items()}
    first_distance_stats = {k: v for k, v in first_distance_stats.items()}
    statistics_dict = {
        "recall": recall_results,
        "distance": distance_stats,
        "diff_distance": diff_stats,
        "first_distance": first_distance_stats,
    }
    return statistics_dict
