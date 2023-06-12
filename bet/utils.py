import json
from typing import Dict, List

from torch import Tensor, device
from torch.utils.data import Dataset
from tqdm import tqdm


def select_field(data: List[Dict], key1: str, key2: str = None) -> List:
    """
    Selects a field from a list of dictionaries.

    Args:
        data (List[Dict]): A list of dictionaries.
        key1 (str): The key to select from each dictionary.
        key2 (str, optional): A nested key to select from each dictionary. Defaults to None.

    Returns:
        List: A list containing the selected fields.
    """
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_iterator(dataloader: Dataset, desc: str, n_samples: int = None) -> tqdm:
    """
    Returns an iterator that can be used to iterate over a PyTorch dataloader.

    Args:
        dataloader (Dataset): A PyTorch dataloader.
        desc (str): A description for the tqdm progress bar.
        n_samples (int, optional): The number of samples in the dataset. Defaults to None.

    Returns:
        tqdm: An iterator that can be used to iterate over the dataloader.
    """
    iter_ = tqdm(dataloader, desc=desc)
    if n_samples:
        iter_ = tqdm(dataloader, total=n_samples / dataloader.batch_size)
    return iter_


def load_candidate_dict(fname: str, params: Dict) -> List[Dict]:
    """
    Loads a list of candidate documents from the candidates document.

    Args:
        fname (str): The path to the file containing the candidate documents.
        params (Dict): A dictionary of parameters.

    Returns:
        List[Dict]: A list of candidate documents.
    """
    doc_list = []
    n = 0
    with open(fname, "rt") as f:
        for line in f:
            line = line.rstrip()  # remove trailing whitespace
            item = json.loads(line)
            text = item.get("label", None) or item.get(
                "text", None
            )  # get either label or text
            title = None
            if params.get("data_use_title"):  # check if title should be used
                title = item.get("candidate_title", None)
            candidate_id = item.get("candidate_id", n)  # get candidate id or use n
            doc_list.append(
                {"text": text, "candidate_title": title, "candidate_id": candidate_id}
            )
            n += 1
            if (
                params["training_debug"] and len(doc_list) >= 200
            ):  # break loop if debug mode is on and list length is greater than 200
                break
    return doc_list


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)

    Got from sentence_transformers
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch
