"""
Should use the already parsed (from training) hdf5 dataset withe the tokenized candidates. 
To make it easy to change dataset evaluation will use RetrievalRawCandidatesDataset directly (and to create the code for inference at the same time)

Needs to remove break on candidates reading for debug

"""
import logging
import os

from bet.bet_parser import BetParser
from bet.text.eval import full_eval

logger = logging.getLogger("bet")


if __name__ == "__main__":
    parser = BetParser(eval=True)
    params = parser.parse_groups()

    output_path = params["output_path"]
    os.makedirs(params["output_path"], exist_ok=True)
    full_eval(params, "test")