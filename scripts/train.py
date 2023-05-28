import os
import logging

from bet.text.train_model import train_biencoder
from bet.bet_parser import BetParser

logger = logging.getLogger("bet")

if __name__ == "__main__":
    parser = BetParser(training=True, eval=True)
    params = parser.parse_groups()
    if params["training_debug"]:
        params["output_path"] = os.path.join(params["output_path"], "training_debug")
        params["data_cache_path"] = os.path.join(params["data_cache_path"], "training_debug")
        params["training_val_check_interval"] = 0.5
        params["testing_batch_size"] = 8
        params["testing_eval_recall"] = 1
        params["training_auto_batch_size"] = False
        params["training_batch_size"] = 8
    os.makedirs(params["output_path"], exist_ok=True)
    #

    train_biencoder(params)