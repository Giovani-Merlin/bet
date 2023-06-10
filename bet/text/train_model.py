"""

"""

import json
import logging
import os
import random
from typing import Dict

import lightning as pl
import numpy as np
import tables
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

from bet.text.datasets.retrieval_raw_dataset import RetrievalDataModule
from bet.text.model.trainer import TextBiEncoderTrainer

logger = logging.getLogger("bet")


def train_biencoder(params: Dict[str, str]):
    # Init trainer
    # Set precision if tensor cores are available
    torch.set_float32_matmul_precision("medium")
    data_module = RetrievalDataModule(params)
    text_bi_encoder_trainer = TextBiEncoderTrainer(
        training_params=params,
    )

    seed = params["training_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_base_path = params["output_path"]
    recall_metric_name = f"recall_R@{params['testing_eval_recall']}"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor=f"{recall_metric_name}",
        mode="max",
        dirpath=os.path.join(output_base_path, "checkpoints/"),
        filename=f"text-biencoder-{{epoch:02d}}-{{{recall_metric_name}:.4f}}",
    )
    early_stop_callback = EarlyStopping(
        monitor=f"{recall_metric_name}",
        min_delta=0.00,
        patience=params["training_patience"],
        verbose=True,
        mode="max",
    )
    # Get tb logger
    tb_logger = TensorBoardLogger(output_base_path, name="lightning_logs")
    # Inject checkpoint inside log dir - to automatically save checkpoints inside versioned dir
    checkpoint_dir = tb_logger.log_dir
    checkpoint_callback.dirpath = checkpoint_dir

    trainer = pl.Trainer(
        # gpus=train_config["train_gpus"],
        max_epochs=params["training_max_epochs"],
        devices=params.get(
            "training_devices", "auto"
        ),  # TODO: Always auto - needs to add on arg parser
        accelerator=params.get("training_accelerator", "auto"),
        min_epochs=params["training_min_epochs"],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        default_root_dir=output_base_path,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=tb_logger,
        val_check_interval=params["training_val_check_interval"],
        precision=params.get("training_precision", "32"),
        # strategy="deepspeed_stage_2_offload",
    )
    try:
        if params.get("training_auto_batch_size"):
            tuner = Tuner(trainer)
            tuner.scale_batch_size(
                text_bi_encoder_trainer, mode="binsearch", datamodule=data_module
            )
            # Reduce 5% of the batch size to avoid OOM - tuner optimizes too much
            data_module.batch_size = int(data_module.batch_size * 0.95)
            # make it divisible by 8 to use tensor cores
            data_module.batch_size = data_module.batch_size - data_module.batch_size % 8
            params["training_batch_size"] = data_module.batch_size
            # Avoir error when saving hparams, needs to have the same keys for the model and the datamodule
            data_module.params["training_batch_size"] = data_module.batch_size
            logger.info("Effective optimized batch size: %d", data_module.batch_size)
        # First validation to check raw performance
        trainer.validate(text_bi_encoder_trainer, datamodule=data_module)
        # Also test to check raw performance
        trainer.test(text_bi_encoder_trainer, datamodule=data_module)
        logger.info(f"Saving model on {checkpoint_dir}")
        trainer.profile = "simple"
        trainer.fit(
            model=text_bi_encoder_trainer,
            datamodule=data_module
            # ckpt_path=self.train_config.get("train_continue_from_checkpoint", None),
        )
    # Keyboard, OOM, etc
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        # Avoid corrupted hdf5 files and save best checkpoint as final model on output dir
        tables.file._open_files.close_all()

    # Save training params
    with open(os.path.join(checkpoint_dir, "training_params.json"), "w") as outfile:
        json.dump(params, outfile)

    # Load best checkpoint
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    logger.info("Loading best checkpoint: %s", best_checkpoint_path)
    text_bi_encoder_trainer = TextBiEncoderTrainer.load_from_checkpoint(
        best_checkpoint_path
    )
    # Save candidate and query final encoders
    text_bi_encoder_trainer.candidate_encoder.save_model(checkpoint_dir)
    text_bi_encoder_trainer.query_encoder.save_model(checkpoint_dir)
    logger.info("Final candidate/query encoder saved to: %s", checkpoint_dir)
    # Check/save results
    validation_results = trainer.validate(
        text_bi_encoder_trainer, datamodule=data_module
    )
    test_results = trainer.test(text_bi_encoder_trainer, datamodule=data_module)
    # Save results
    results = {
        "validation": validation_results,
        "test": test_results,
    }
    with open(os.path.join(checkpoint_dir, "training_results.json"), "w") as outfile:
        json.dump(results, outfile)
