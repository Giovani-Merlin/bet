# pylint: disable=no-member
# pylint: disable=not-callable
import json
import logging
import os
from typing import Dict, List, Union

import numpy as np
import torch
from tqdm.autonotebook import trange
from transformers import AutoConfig, AutoModel, AutoTokenizer

from bet.text.datasets.constants import ENT_END_TAG, ENT_START_TAG, ENT_TITLE_TAG
from bet.text.vector_store import INDEX_NAME
from bet.text.vector_store.scann_ids import ScannIds

CONFIG_NAME, WEIGHTS_NAME = "config.json", "model.bin"

logger = logging.getLogger("bet")

from bet.text.datasets.utils import (
    get_candidate_representation,
    get_query_representation,
)


####
class BaseEncoder(torch.nn.Module):
    """
    It is a wrapper around a transformer model, handling the model loading and the forward pass.
    It build the model by:
        Loading a HuggingFace`s pre-trained model if the parameter "model" is not provided - to allow using custom models.
        Customizing the model by reducing the number of layers - using of "Encoder/Decoder_num_hidden_layers" parameter
        Adding a second model on top of the model - using of "Encoder/Decoder_add_model" parameter
            ATM only "linear" is supported
    It modifies the forward pass by:
        ! TODO(GM): modify this:
        Uses by default the cls token as the output of the model
        Normalizes the output of the model to use cos distance

    # ! TODO(GM): Apart from the index functions, it has a full match with sentence bert... Just need to adapt optimizer
    # Just use it as a base class and add a new level of abstraction for the index functions....

    if tokenizer resize the input to the max length of the model
    """

    def __init__(
        self,
        params: dict,
        model_type: str,
    ):
        super().__init__()
        self.params = params
        self.model_type = model_type
        self.model = None
        self.init_model(params, model_type)
        # Search attributes
        self.index = None

    def init_model(self, params: dict, model_type: str):
        """Initializes the model."""
        config = AutoConfig.from_pretrained(params[f"{self.model_type}_model"])

        self.model = AutoModel.from_pretrained(
            params[f"{self.model_type}_model"], config=config
        )
        # ! TODO(GM): Make it candidate/query specific argument
        if params.get(f"training_reset_last_n_layers", None):
            # Reset last n layers
            reset_last_n_layers = params[f"training_reset_last_n_layers"]
            # Some models are called transformer and others encoder
            if hasattr(self.model.base_model, "transformer"):
                encoder_model = self.model.base_model.transformer
            elif hasattr(self.model.base_model, "encoder"):
                encoder_model = self.model.base_model.encoder
            for i in range(reset_last_n_layers):
                encoder_model.layer[-1 - i].apply(self.model.base_model._init_weights)

        # ! Add special head
        # ! TODO(GM): Make it useful by changing to "base_model" to use siamese structure and just different over_models
        output_dim = self.model.base_model.config.hidden_size
        append_model = self.params.get(f"{model_type}_append_model", None)
        if append_model:
            if append_model == "ffn":
                self.over_model = torch.nn.Sequential(
                    torch.nn.Linear(output_dim, output_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(
                        output_dim, self.params[f"{model_type}_output_dimension"]
                    ),
                )
        else:
            self.over_model = None
        #
        self.params = params
        if self.over_model:
            # Injects over the encoder to get parameters with optimizer
            self.model = torch.nn.Sequential(
                self.model.base_model.encoder, self.over_model
            )

    def forward(self, model_input: dict):
        """
        Forward pass of the model.

        model_input is obtained by using "tokenize_plus" method of the tokenizer - named parameters.
        """

        output = self.model.base_model(**model_input)
        # TODO(GM): Use or not use pooler output? If yes, needs to add on optimizer
        last_hidden_state = output["last_hidden_state"]
        cls_output = last_hidden_state[:, 0]
        # Normalizes the model with numerical stability
        eps = 1e-8
        embeddings_n = cls_output.norm(dim=1)[:, None]
        embeddings = cls_output / torch.clamp(embeddings_n, min=eps)

        return embeddings

    def save_model(self, model_path: str = None):
        """Saves the model in the output directory."""
        if not model_path:
            model_path = self.params["output_path"]

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        output_model_file = os.path.join(model_path, self.name + "_" + WEIGHTS_NAME)
        torch.save(self.state_dict(), output_model_file)
        output_config_file = os.path.join(model_path, self.name + "_" + CONFIG_NAME)
        with open(output_config_file, "w") as outfile:
            json.dump(self.params, outfile)

    def load_weights(self, model_path=None, device="cpu"):
        model_path = os.path.join(model_path, self.name + "_" + WEIGHTS_NAME)

        try:
            state_dict = torch.load(model_path, map_location=device)
            logger.info(f"Model weight's loaded from {model_path}")
            self.load_state_dict(state_dict, strict=False)
            self.to(device)
        except FileNotFoundError:
            logger.warning(
                f"Model weight's not found on {model_path}, starting model with random"
                " weights"
            )

    @classmethod
    def load_model(cls, model_path, model_name=None, device="cpu"):
        """Loads the model from the output directory."""

        output_config_file = os.path.join(model_path, model_name + "_" + CONFIG_NAME)
        with open(output_config_file, "r") as cfile:
            params = json.load(cfile)
            params["output_path"] = model_path

        model = cls(params)
        model.load_weights(
            model_path,
            device,
        )
        #       if "16" in params["training_precision"]:
        #       model = model.half()
        return model

    # Functions for searching
    def create_index(
        self,
        sentences: List[str] = None,
        encoded_sentences=None,
        device: str = "cpu",
        batch_size: int = 32,
        index_ids=None,
        index_type: str = "scann",
        index_configs: dict = {},
        index_to_title: dict = None,
    ):
        """
        Creates an index with the sentences passed as argument.

        Args:
            sentences: A list of sentences to index
            device: Device to use for the encoding process

        Returns:
            Faiss index with the sentences encoded

        """
        if encoded_sentences is None:
            encoded_sentences = self.encode(
                sentences, device=device, return_numpy=True, batch_size=batch_size
            )
        if index_type == "scann":
            index = ScannIds(
                ids=index_ids,
                dataset=encoded_sentences,
                index_configs=index_configs,
                index_to_title=index_to_title,
            )
        self.index = index
        return index

    def save_index(self, index_path: str = None):
        """Saves the index in the output directory."""
        if not index_path:
            index_path = self.params["output_path"]
            index_path = os.path.join(index_path, INDEX_NAME)

        if not os.path.exists(index_path):
            os.makedirs(index_path)

        self.index.save_index(index_path)

    def load_index(
        self,
        index_path: str = None,
        index_type: str = "scann",
    ):
        """Loads the index from the output directory."""
        if not index_path:
            index_path = self.params["output_path"]
            index_path = os.path.join(index_path, INDEX_NAME)

        if os.path.exists(index_path):
            if index_type == "scann":
                index = ScannIds.load_pretrained(index_path)
                self.index = index
                return index

        else:
            logger.warning(f"Index not found on {index_path}")
            return None

    def search(
        self,
        query: str = None,
        encoded_query: np.ndarray = None,
        top_k: int = 10,
        index=None,
        device: str = "cpu",
        as_titles: bool = False,
    ):
        """
        Search for similar sentences.

        Args:
            query: A single sentence or a list of sentences to use as query
            top_k: Number of top similar sentences to retrieve
            index: Faiss index to use for the search
            encoded_query: Encoded query to use for the search

        Returns:
            A list of tuples (score, sentence)

        """
        if encoded_query is None:
            encoded_query = self.encode(query, return_numpy=True, device=device)

        if index is None:
            index = self.index
        if len(encoded_query.shape) == 1:
            encoded_query = encoded_query.reshape(1, -1)
        indexes, scores = index.search(encoded_query, top_k, as_titles=as_titles)
        return indexes, scores


#
class CandidateEncoder(BaseEncoder):
    def __init__(self, params):
        self.name = "candidate_encoder"
        super(CandidateEncoder, self).__init__(params, self.name)
        # Candidate has one more token - [title]
        self.tokenizer = self.get_tokenizer()
        self.model.resize_token_embeddings(len(self.tokenizer.get_vocab()))

    @classmethod
    def load_model(cls, model_path, device="cpu"):
        """Loads the model from the output directory."""
        return super().load_model(model_path, "candidate_encoder", device)

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.params["query_encoder_model"])
        tokenizer.add_special_tokens({"additional_special_tokens": [ENT_TITLE_TAG]})
        return tokenizer

    def encode(
        self,
        candidates: Union[List[str], str],
        device: str = "cpu",
        batch_size: int = 32,
        return_numpy: bool = True,
    ):
        """
        TODO(GM): Needs to adapt it like sentence_transformers to group by size
        Encode candidates using the model.

        Args:
            candidates: A list of or a single candidate to encode
            device: Device to use for the encoding process
            batch_size: Batch size to use for the encoding process
            return_numpy: Whether to return a numpy array or a tensor - if numpy always returns on cpu, if tensor returns on the device

        Returns:
            A numpy array or a tensor with the embeddings

        """
        candidates = [candidates] if isinstance(candidates, str) else candidates
        # Like sentence transformers, sort by length
        length_sorted_idx = np.argsort([-len(sen) for sen in candidates])
        candidates_sorted = [candidates[idx] for idx in length_sorted_idx]
        all_features = []
        for start_index in trange(0, len(candidates), batch_size, desc="Batches"):
            candidates_batch = candidates_sorted[start_index : start_index + batch_size]
            # Encode batch
            encoded_candidates_batch = get_candidate_representation(
                candidates_batch,
                self.tokenizer,
                self.params["data_candidate_max_length"],
                self.params.get("data_candidate_title"),
                return_tensors="pt",
                padding="longest",
            )

            # Put on device
            encoded_candidates_batch = {
                key: value.to(device) for key, value in encoded_candidates_batch.items()
            }

            with torch.no_grad():
                out_features = self.forward(encoded_candidates_batch)
                if return_numpy:
                    # Save precision as fp16 to save space
                    out_features = out_features.cpu().numpy().astype(np.float16)
                all_features.extend(out_features)
        # Return to the original order
        all_features = [all_features[idx] for idx in np.argsort(length_sorted_idx)]

        return np.array(all_features) if return_numpy else torch.tensor(all_features)


class QueryEncoder(BaseEncoder):
    def __init__(self, params):
        self.name = "query_encoder"
        super(QueryEncoder, self).__init__(params, self.name)
        # Query has 2 more tokens - [begin of query] and [end of query]
        self.tokenizer = self.get_tokenizer()
        self.model.resize_token_embeddings(len(self.tokenizer.get_vocab()))

    @classmethod
    def load_model(cls, model_path, device="cpu"):
        """Loads the model from the output directory."""
        return super().load_model(model_path, "query_encoder", device)

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.params["query_encoder_model"])
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [ENT_START_TAG, ENT_END_TAG]}
        )
        return tokenizer

    def encode(
        self,
        queries: Union[Dict[str, str], List[Dict[str, str]]],
        device: str = "cpu",
        batch_size: int = 32,
        return_numpy: bool = True,
    ):
        """
        TODO(GM): Needs to adapt it like sentence_transformers to group by size
        Encode queries using the model.

        Args:
            queries: A list of or a single query to encode
            device: Device to use for the encoding process
            batch_size: Batch size to use for the encoding process
            return_numpy: Whether to return a numpy array or a tensor - if numpy always returns on cpu, if tensor returns on the device

        Returns:
            A numpy array or a tensor with the embeddings

        """
        queries = [queries] if isinstance(queries, str) else queries
        # Like sentence transformers, sort by length
        length_sorted_idx = np.argsort([-len(sen) for sen in queries])
        queries_sorted = [queries[idx] for idx in length_sorted_idx]
        all_features = []
        for start_index in trange(0, len(queries), batch_size, desc="Batches"):
            queries_batch = queries_sorted[start_index : start_index + batch_size]
            # Encode batch
            encoded_queries_batch = get_query_representation(
                queries_batch,
                self.tokenizer,
                max_seq_length=self.params["data_query_max_length"],
                return_tensors="pt",
                padding="longest",
            )

            # Put on device
            encoded_queries_batch = {
                key: value.to(device) for key, value in encoded_queries_batch.items()
            }

            with torch.no_grad():
                out_features = self.forward(encoded_queries_batch)
                if return_numpy:
                    out_features = out_features.cpu().numpy()
                all_features.extend(out_features)
        # Return to the original order
        all_features = [all_features[idx] for idx in np.argsort(length_sorted_idx)]
        return np.array(all_features) if return_numpy else torch.tensor(all_features)
