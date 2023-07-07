import argparse
import os
from datetime import datetime

import yaml


def date_time_arg(date_time_str):
    """Convert date time argument to datetime object"""
    if date_time_str:
        return datetime.strptime(date_time_str, "%Y-%m-%d")


def none_or_str(value):
    if value == "None":
        return None
    return value


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True" or s == "true"


class BaseParser(argparse.ArgumentParser):
    """
    Base parser for all the parsers, it implements the following:

    - Parse defaults from a yaml file
    - Group arguments
    """

    def __init__(self, *args, **kwargs):
        super(BaseParser, self).__init__(*args, **kwargs)

    def parse_groups(self, args=None, namespace=None):
        """
        Helper function to parse the arguments divided by groups.
        Also, it reads arguments from a json file if the {group_name}_args_file is defined.

        Used for arguments organization. For example, we have a group called "data_processing",
        ones can easily call the text preprocessing function with text_preprocessing(**parsed_arguments["data_processing"])
        """
        args = super().parse_known_args(args, namespace)
        print(f"Unknown args {args[1]}")
        args_per_group = self.get_args_per_group(args[0])
        # Make flat dictionary with all the groups
        return {k: v for d in args_per_group.values() for k, v in d.items()}

    def get_args_per_group(self, args):
        """
        Helper function to get the arguments of a group.
        Also verify mutually exclusive arguments.
        """
        arg_groups = {}

        for group in self._action_groups:
            # Group args
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            # Check if there are mutually exclusive arguments
            exclusive_groups = [
                {a.dest: getattr(args, a.dest, None) for a in a._group_actions}
                for a in group._mutually_exclusive_groups
            ]
            # If there are, check if there are more than one argument defined
            for exclusive_group in exclusive_groups:
                if sum([1 for v in exclusive_group.values() if v is not None]) > 1:
                    raise ValueError(f"Mutually exclusive arguments {exclusive_group.keys()} are not allowed")
            arg_groups[group.title] = group_dict
        return arg_groups


class BetParser(BaseParser):

    """Arg parser class for BET training model"""

    def __init__(
        self,
        defaults_path="config/text/retrieval.yaml",
        training=True,
        eval=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            description="BET - Bi-Encoder Toolkit for Text Retrieval",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.bet_home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # Load defaults from yaml file
        # This allow us to check the defaults with the --help command
        # and also to change the arguments by sending them as cli arguments
        # Format is --{arg_category}_{arg_name} {arg_value}
        self.training = training
        self.eval = eval
        self.model_defaults = yaml.safe_load(
            open(
                defaults_path,
                "r",
                encoding="utf-8",
            )
        )
        self.add_argument(
            "--output_path",
            type=str,
            default=self.model_defaults["output_path"],
            help="Output path to save any script output",
            metavar="\b",
        )
        self.add_data_args()
        self.add_model_args()
        if self.training:
            self.add_training_args()
        if self.eval:
            self.add_testing_args()

    def add_data_args(self):
        """
        To define the data to be used.
        """
        arg_category = "data"
        group = self.add_argument_group(arg_category)
        group.add_argument(
            f"--{arg_category}_data_path",
            type=str,
            default=self.model_defaults[arg_category]["data_path"],
            help="Path to the training data as BET format",
        )
        group.add_argument(
            f"--{arg_category}_cache_path",
            type=str,
            default=self.model_defaults[arg_category]["cache_path"],
            help="Path to the hdf5 data files",
        )
        group.add_argument(
            f"--{arg_category}_use_title",
            type=boolean_string,
            default=self.model_defaults[arg_category]["use_title"],
            help="Whether to use the title of the document before the text",
        )
        group.add_argument(
            f"--{arg_category}_workers",
            type=int,
            default=self.model_defaults[arg_category]["workers"],
            help="Number of workers to use for data loading",
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_candidate_max_length",
            type=int,
            default=self.model_defaults[arg_category]["candidate_max_length"],
            help="Max length of the text to be used",
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_query_max_length",
            type=int,
            default=self.model_defaults[arg_category]["query_max_length"],
            help="Max length of the text to be used",
            metavar="\b",
        )

    def add_model_args(self):
        """
        To be used to chose the model architecture
        """
        arg_category = "models"
        group = self.add_argument_group(arg_category)
        self.add_encoder_args(group, self.model_defaults[arg_category]["query_encoder"], "query_encoder")
        self.add_encoder_args(
            group,
            self.model_defaults[arg_category]["candidate_encoder"],
            "candidate_encoder",
        )

    def add_encoder_args(self, group, defaults, arg_category):
        """
        To be used to chose the candidate or query model architecture
        """
        group.add_argument(
            f"--{arg_category}_model",
            type=str,
            default=defaults["model"],
            help="Hugging face model to use. Tested for bert, xlm, distilbert, roberta and distilroberta",
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_append_model",
            type=str,
            default=defaults["append_model"],
            help="Model to concatenate on the output",
            metavar="\b",
            choices=["ffn"],
        )
        group.add_argument(
            f"--{arg_category}_output_dimension",
            type=str,
            default=defaults["output_dimension"],
            help="Only valid when append a model",
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_weights_path",
            type=str,
            default=defaults["weights_path"],
            help="Path to the weights to be loaded - if not specified, the model will use the pretrained weights of the hugginface model",
            metavar="\b",
        )

    def add_training_args(self):
        """
        To be used for training - to train the model.
        """
        arg_category = "training"
        group = self.add_argument_group(arg_category)
        group.add_argument(
            f"--{arg_category}_debug",
            type=boolean_string,
            default=self.model_defaults[arg_category]["debug"],
            help="Whether to use the title of the document before the text",
        )
        group.add_argument(
            f"--{arg_category}_seed",
            type=int,
            default=self.model_defaults[arg_category]["seed"],
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_shuffle",
            type=boolean_string,
            default=self.model_defaults[arg_category]["shuffle"],
            metavar="\b",
        )

        group.add_argument(
            f"--{arg_category}_random_negatives_loss_scaler",
            type=float,
            default=self.model_defaults[arg_category]["random_negatives_loss_scaler"],
            help="Random negatives loss scaler initial value - to transform cos similarity in logits",
        )

        group.add_argument(
            f"--{arg_category}_learning_rate",
            type=float,
            default=self.model_defaults[arg_category]["learning_rate"],
            metavar="\b",
        )

        group.add_argument(
            f"--{arg_category}_batch_size",
            type=int,
            default=self.model_defaults[arg_category]["batch_size"],
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_auto_batch_size",
            type=boolean_string,
            default=self.model_defaults[arg_category]["auto_batch_size"],
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_patience",
            type=int,
            default=self.model_defaults[arg_category]["patience"],
            help="Patience for early stopping - units of validation check interval",
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_val_check_interval",
            type=float,
            default=self.model_defaults[arg_category]["val_check_interval"],
            help="Validation check interval - units of batches, also defines the patience for early stopping",
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_precision",
            type=str,
            default=self.model_defaults[arg_category]["precision"],
            help="Precision to use for training as lightning str format - 16 or 32",
            metavar="\b",
        )

        group.add_argument(
            f"--{arg_category}_weight_decay",
            type=float,
            default=self.model_defaults[arg_category]["weight_decay"],
            help="Weight decay for optimizer",
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_warmup_proportion",
            type=float,
            default=self.model_defaults[arg_category]["warmup_proportion"],
            help="Warmup proportion for optimizer",
            metavar="\b",
        )

        group.add_argument(
            f"--{arg_category}_max_epochs",
            type=int,
            default=self.model_defaults[arg_category]["max_epochs"],
            help="Max number of epochs to perform the training",
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_min_epochs",
            type=int,
            default=self.model_defaults[arg_category]["min_epochs"],
            help="Min epochs to perform before stopping the training",
            metavar="\b",
        )

        group.add_argument(
            f"--{arg_category}_continue_from_checkpoint",
            type=str,
            default=None,
            help="Path to the checkpoint to continue training from",
            metavar="\b",
        )

        group.add_argument(
            f"--{arg_category}_metric_tracking",
            default=self.model_defaults[arg_category]["metric_tracking"],
            type=int,
            help="Metric to monitor in training. Checkpoint and early stopping will use this metric. For recall use recall_R@x where x is the number of candidates to search for (e.g. recall_R@5)",
            metavar="\b",
        )
        group.add_argument(
            f"--{arg_category}_metric_tracking_mode",
            type=str,
            default=self.model_defaults[arg_category]["metric_tracking_mode"],
            metavar="\b",
            help="Metric mode to use in training. Modes are min or max",
        )
    def add_testing_args(self):
        """
        To be used for testing - to evaluate model performance and/or to optimize hyperparameters.
        """
        arg_category = "testing"
        group = self.add_argument_group(arg_category)
        group.add_argument(
            f"--{arg_category}_batch_size",
            type=int,
            default=self.model_defaults[arg_category]["batch_size"],
            metavar="\b",
        )

        group.add_argument(
            f"--{arg_category}_top_k",
            default=self.model_defaults[arg_category]["top_k"],
            type=int,
            help="Number of candidates to search for (and compute statistics) when doing full evaluation",
            metavar="\b",
        )

        group.add_argument(
            f"--{arg_category}_index_path",
            default=self.model_defaults[arg_category]["index_path"],
            type=str,
            help="Path to the index to be loaded/created",
            metavar="\b",
        )
