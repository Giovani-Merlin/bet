# pylint: disable=no-member
# pylint: disable=not-callable
import logging
from operator import attrgetter

import lightning as pl
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from bet.text.model.model import CandidateEncoder, QueryEncoder

logger = logging.getLogger(__name__)


# define the LightningModule
class TextBiEncoderTrainer(pl.LightningModule):
    def __init__(self, training_params: dict):
        super().__init__()
        self.params = training_params
        self.save_hyperparameters()
        if not training_params["candidate_encoder_weights_path"]:
            self.candidate_encoder = CandidateEncoder(training_params)
        else:
            self.candidate_encoder = CandidateEncoder.load_model(
                training_params["candidate_encoder_weights_path"]
            )
        if not training_params["query_encoder_weights_path"]:
            self.query_encoder = QueryEncoder(training_params)
        else:
            self.query_encoder = QueryEncoder.load_model(
                training_params["query_encoder_weights_path"]
            )

        self.random_negatives_loss_scaler = torch.nn.Parameter(
            torch.tensor(
                float(training_params["training_random_negatives_loss_scaler"]),
                requires_grad=True,
            )
        )
        # Used to do better evaluation - by using all the candidates as the pool
        self.candidates_eval = []
        self.val_output = []
        #

    def score_candidates(
        self,
        batch,
        margin: float = 0.2,
    ):
        """
        Scores the candidates for each query vector
        """
        query_inputs, candidate_inputs, auxiliar = batch

        query_vecs = self.query_encoder(query_inputs)
        candidate_vecs = self.candidate_encoder(candidate_inputs)
        scores = query_vecs.mm(candidate_vecs.t())
        # If we have repeated candidates in the batch we need to mask as 0 the scores for the repeated candidates
        candidates_idx = auxiliar["candidate_index"].ravel()
        # Make symmetric matrix putting 1 in the same elements position and 0 in the rest
        candidates_mask = np.equal.outer(candidates_idx, candidates_idx)
        # Invert the diagonal to keep it
        candidates_mask[np.diag_indices_from(candidates_mask)] = ~candidates_mask[
            np.diag_indices_from(candidates_mask)
        ]
        # -100 scores on it
        scores.view(-1)[candidates_mask.ravel()] = -100
        # Research about how the candidates are being grouped
        # ! TODO: SHOULD Be DONE WITH CALLBACK (clean code) - but would need to do query vecs and candidate vecs outside score_candidates - needs to check it better
        candidates_mask = ~np.equal.outer(candidates_idx, candidates_idx)
        candidates_similarities = candidate_vecs @ candidate_vecs.T
        # Clip the candidate similarity to be at least the margin - negative values will be treated as 0
        candidates_similarities = torch.clip(candidates_similarities - margin, min=0)
        mean_candidate_similarity = candidates_similarities.view(-1)[
            candidates_mask.ravel()
        ].mean()
        self.log_dict(
            {"mean_candidate_similarity": mean_candidate_similarity},
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        self.log_dict(
            {"random_negatives_loss_scaler": self.random_negatives_loss_scaler},
            on_step=True,
            prog_bar=True,
            logger=True,
        )

        return scores * self.random_negatives_loss_scaler, mean_candidate_similarity

    def training_step(self, batch, batch_idx):
        scores, mean_candidate_similarity = self.score_candidates(
            batch, margin=self.params["training_loss_candidates_margin"]
        )

        # Doing the matrix multiplication we have the correct index at the diagonal position and all the other positions are the in batch negatives examples
        # as it uses softmax implicitly - We use the dot product as our scoring function
        # Explained here: https://arxiv.org/pdf/1911.03814.pdf - section 4.1
        # Measure is R@1
        # Loss with logits
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        labels = torch.arange(scores.shape[0], device=self.device)
        query_candidate_loss = cross_entropy_loss(scores, labels)
        # Needs to think about this class. Can be BCELOSS loss (classification to be 0, max inf min 0), mse loss (max 1 min 0), etc
        # ! LAST TEST - says that if 0.2 is the mean similarity it's good (hyperparameter) - As it is impossible to have 0 distance between all vectors if n_candidates > n_dim
        # This param should be optimized given the size of the candidates and the output of the encoder. This plays directly with the scaling of the scores
        # If scaling is high the query can be just +- close to the candidate and probably the candidates are closer to each other and therefore the mean_candidate_similarity is high (requires higher margin).
        # If Scaling is low and we have no margin the mean_candidate_similarity is low but the cross entropy loss will increase as it will be hard to find the correct candidate in a totally sparse space
        # Choice now is to accept a heuristic margin (accept candidates to be max 0.2 distance from each other).
        loss = query_candidate_loss
        if self.params["training_loss_candidates_similarity"]:
            loss += mean_candidate_similarity
        self.log_dict(
            {
                "train_loss": loss,
                "query_candidate_loss": query_candidate_loss,
                "scores": torch.diagonal(scores).mean(),
            },
            on_step=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "lr_",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.query_encoder, norm_type=2)
    #     self.log_dict(norms)

    def validation_step(self, batch, batch_idx):
        # Just encode candidates
        query_inputs, candidates_inputs, auxiliar = batch
        candidates_idx = auxiliar["candidate_index"].ravel()
        candidates_to_encode = []
        # We will encode just one time each candidate
        # For that we will check each candidate index and if it's not already encoded we will add all its inputs to the list
        candidates_values = list(candidates_inputs.values())
        for n, candidate_index in enumerate(candidates_idx):
            # If it's not already encoded
            if candidate_index not in self.candidates_eval:
                candidates_to_encode.append(
                    [candidate_input[n] for candidate_input in candidates_values]
                )
                self.candidates_eval.append(candidate_index)
        # Remap batch encode plus with the candidates
        # For that we will zip the inputs (transpose) and then stack them
        candidates = None
        if len(candidates_to_encode) > 0:
            candidates = self.candidate_encoder(
                {
                    key: torch.vstack(value)
                    for key, value in zip(
                        candidates_inputs.keys(), list(zip(*candidates_to_encode))
                    )
                }
            )
        query = self.query_encoder(query_inputs)
        val_output = {
            "query_encoded": query.detach().cpu(),
            "candidates_idx": candidates_idx,
            "candidates_encoded": candidates.detach().cpu()
            if candidates is not None
            else None,
        }
        self.val_output.append(val_output)

    def test_step(self, batch, batch_idx):
        # Do equal to validation step
        self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # From validation step we have:
        # Query encoded and candidates_idx to know which candidate for each query
        # candidates_encoded - all the candidates encoded
        # self.canidates_eval - all the candidates idx that were encoded (mapping to candidates_encoded)
        outputs = self.val_output
        all_candidates = torch.vstack(
            [
                output["candidates_encoded"]
                for output in outputs
                if output["candidates_encoded"] is not None
            ]
        )
        all_queries = torch.vstack([output["query_encoded"] for output in outputs])
        all_correct_candidates_idx = [
            candidate_index
            for batch_candidates_idx in outputs
            for candidate_index in batch_candidates_idx["candidates_idx"]
        ]
        # batch vector multiplication
        scores = torch.bmm(
            all_queries.unsqueeze(0), all_candidates.t().unsqueeze(0)
        ).squeeze()
        # Sort results by candidate idx
        ordered_results = scores.cpu().argsort(dim=1, descending=True)
        # Map candidate idx to position in candidates_pool
        query_to_candidate_map = [
            self.candidates_eval.index(idx) for idx in all_correct_candidates_idx
        ]
        # Get the position of the correct candidate in the ordered results
        correct_candidate_position = [
            np.where(ordered_result == candidate_index)[0][0]
            for ordered_result, candidate_index in zip(
                ordered_results, query_to_candidate_map
            )
        ]
        # Get de testing_eval_recall score
        recall_ref = self.params["testing_eval_recall"]
        score = np.mean(
            [
                1 if position < recall_ref else 0
                for position in correct_candidate_position
            ]
        )
        self.candidates_eval.clear()  # Reset candidates_eval
        self.val_output.clear()
        self.log(f"recall_R@{recall_ref}", score, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        # Do equal to validation epoch end
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        """
        Returns specialized optimizer for the model and optimization strategy.
        """
        # To avoid problems with different model names like bert, ffn, etc we will use the base_model default from huggingface
        # Currently we support optimizing for all layers (embeddings + encoder) or just the encoder layers
        if self.params["training_optimization_strategy"] == "all":
            model_to_optimize = [
                "query_encoder.model.base_model",
                "candidate_encoder.model.base_model",
            ]
        elif self.params["training_optimization_strategy"] == "encoder_layers":
            model_to_optimize = [
                "query_encoder.model.base_model.encoder",
                "candidate_encoder.model.base_model.encoder",
            ]

        parameters_with_decay = []
        parameters_with_decay_names = []
        parameters_without_decay = []
        parameters_without_decay_names = []
        not_optimized_parameters_names = []
        # We will not decay the bias, gamma and beta parameters
        no_decay = ["bias", "gamma", "beta"]

        for model in [attrgetter(model_name)(self) for model_name in model_to_optimize]:
            for n, p in model.named_parameters():
                if any(t in n for t in no_decay):
                    parameters_without_decay.append(p)
                    parameters_without_decay_names.append(n)
                else:
                    parameters_with_decay.append(p)
                    parameters_with_decay_names.append(n)
            else:
                p.requires_grad = False
                not_optimized_parameters_names.append(n)

        logger.info(f"Model have {len(list(self.named_parameters()))} parameters")
        logger.info(
            f"{len(parameters_with_decay_names)} parameters will be optimized WITH decay"
        )
        logger.info(
            f"{len(parameters_without_decay_names)} parameters will be optimized WITHOUT decay"
        )
        # Like self.random_negatives_loss
        loss_parameters = [self.random_negatives_loss_scaler]
        optimizer_grouped_parameters = [
            {
                "params": parameters_with_decay,
                "training_weight_decay": self.params["training_weight_decay"],
                "lr": self.params["training_learning_rate"],
            },
            {
                "params": parameters_without_decay,
                "training_weight_decay": 0.0,
                "lr": self.params["training_learning_rate"],
            },
        ]
        optimizer_loss_parameters = [
            {  # ! TODO: Use a different optimizer maybe without scheduler
                "params": loss_parameters,
                "training_weight_decay": 0.0,
                "lr": 0.1,  # Make it as hyperparameter...
            }
        ]
        # Needs to do manual optimization to use multiple schedulers/optims...
        # loss_parameters_optimizer = AdamW(optimizer_loss_parameters)
        # scheduler_loss = OneCycleLR(
        #     loss_parameters_optimizer,
        #     max_lr=1,
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     pct_start=self.params["training_warmup_proportion"],
        #     anneal_strategy="linear",
        # )
        optimizer_grouped_parameters = (
            optimizer_grouped_parameters + optimizer_loss_parameters
        )

        optimizer_scheduled = AdamW(optimizer_grouped_parameters)
        # optimizer_loss = AdamW(optimizer_loss_parameters)
        # optimizer_scheduled = NAdam(optimizer_grouped_parameters, eps=1e-8)
        # We will use OneCycleLR to train the model - warmup and decay
        scheduler = OneCycleLR(
            optimizer_scheduled,
            max_lr=self.params["training_learning_rate"],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.params["training_warmup_proportion"],
            anneal_strategy="linear",
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer_scheduled, "lr_scheduler": lr_scheduler_config}
