"""
    Scann is the fastest algorithm, BUT we can't just add new vectors to the index, we need to rebuild it from scratch.
    We can easily use multiple indexes, but it's more work
    Therefore, we migrate to FAISS or Milvus
    
    UPDATE: Faiss has tons of configs... Better just scann and it's good
"""

import os

import scann
import numpy as np
from typing import Dict, Any, List
from scann.scann_ops.py import scann_ops_pybind_backcompat


class ScannIds:
    index_name = "scann"

    def __init__(
        self,
        ids: List[str],
        dataset=None,
        index=None,
        index_to_title=None,
        index_configs: Dict[str, Any] = {"brute_force": True},
    ):
        # Assert that we don't have dataset and index
        assert (dataset is None) or (index is None)
        self.ids = ids
        self.ids_map = np.vectorize({i: n for i, n in enumerate(self.ids)}.get)
        self.index_to_title = (
            np.vectorize(index_to_title.get) if index_to_title else None
        )
        self.index = (
            self.generate_index(dataset, **index_configs)
            if dataset is not None
            else index
        )

    # Recomendations:
    # https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
    def generate_index(self, dataset, brute_force=True, **kwargs):
        if brute_force:
            searcher = (
                scann.scann_ops_pybind.builder(dataset, 64, "dot_product")
                .score_brute_force()
                .build()
            )
        else:
            searcher = (
                scann.scann_ops_pybind.builder(dataset, 64, "dot_product")
                .tree(
                    num_leaves=np.sqrt(dataset.shape[0]).astype(int),
                    num_leaves_to_search=200,
                )
                .score_ah(2, anisotropic_quantization_threshold=0.2)
                # .reorder(100)  # Dataset is needed if we want to reorder....
                .build()
            )

        return searcher

    def save_index(self, persistence_path):
        os.makedirs(persistence_path, exist_ok=True)
        self.index.serialize(persistence_path)
        # Save the ids
        np.save(os.path.join(persistence_path, "ids.npy"), self.ids)

    @classmethod
    def load_pretrained(cls, persistence_path):
        # Scann uses "assets" to get the index information
        # If we change the path, it will not work - so we change the assets file to match the index file
        # To update the assets file we can use scann backwards compatibility - it's a hack but it works
        scann_ops_pybind_backcompat.populate_and_save_assets_proto(persistence_path)

        index = scann.scann_ops_pybind.load_searcher(persistence_path)
        ids = np.load(os.path.join(persistence_path, "ids.npy"))
        return cls(ids, index=index)

    def search(self, query, top_k=1000, as_titles=False):
        indexes, distances = self.index.search_batched(query, final_num_neighbors=top_k)
        # Re-map the indexes to the original ids
        indexes = self.ids_map(indexes.astype(int))
        # Split matrix in list of arrays - to make possible to hide invalid values
        indexes = np.split(indexes, indexes.shape[0])
        distances = np.split(distances, distances.shape[0])
        # If index is not big enough (not enough neighbors in the same cluster) we will have nan as the distance
        # We need to exclude them
        for i, (idx, dist) in enumerate(zip(indexes, distances)):
            # Get the indexes of the invalid values
            invalid_indexes = np.where(np.isnan(dist))[1]
            # Remove the invalid values
            indexes[i] = np.delete(idx, invalid_indexes)
            distances[i] = np.delete(dist, invalid_indexes)
        if as_titles:
            indexes = self.index_to_title(indexes)
        return indexes, distances


#
def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size
