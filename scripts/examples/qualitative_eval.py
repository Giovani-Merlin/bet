"""
Quick script to showcase some qualitative examples for german BET

"""

import logging
import os

import numpy as np
import torch

from bet.bet_parser import BetParser
from bet.text.model.model import QueryEncoder
import json
import re

logger = logging.getLogger("bet")


if __name__ == "__main__":
    parser = BetParser(eval=True)
    params = parser.parse_groups()

    output_path = params["output_path"]
    os.makedirs(params["output_path"], exist_ok=True)
    # Expected format:
    # 'mention', 'query_left', 'query_right'
    as_blink_format = []
    blink_german_example_1 = (
        """Die Schönheit von [ENT]Weltkunst[\ENT] entfaltet sich, inspiriert von der antiken Pracht von [ENT]Myra[\ENT] und den bezaubernden Landschaften von [ENT]Ruffec[\ENT]."""
        """ In [ENT]Myra[\ENT], mit seinen beeindruckenden antiken Ruinen, verbindet sich die Vergangenheit mit der Gegenwart. Die malerischen Straßen von [ENT]Ruffec[\ENT] hingegen zeugen"""
        """ von einem charmanten französischen Dorfleben. Diese Orte verschmelzen zu einer kulturellen Symphonie, die die Vielfalt der Welt in einer einzigartigen künstlerischen Form einfängt"""
    )
    blink_german_example_2 = (
        "Tauchen Sie ein in die vielfältige Klangwelt von [ENT]Karussell[\ENT], die"
        " Label in Berlin, die eine Melodie aus Nachrichten und Musik präsentiert. Dabei"
        " versprüht das traditionelle [ENT]Karussell[\ENT] als Nachbildung einer"
        " Pferdesport einen Hauch von nostalgischem Charme. Rockiges"
        " [ENT]Karussell[\ENT] liefert dagegen energiegeladene Rhythmen und mitreißende"
        " Melodien, die eine eigene musikalische Achterbahnfahrt erzeugen."
    )
    blink_examples = [blink_german_example_1, blink_german_example_2]
    for blink_example in blink_examples:
        for match in re.finditer(r"\[ENT\](.*?)\[\\ENT\]", blink_example):
            context_left = blink_example[: match.start()]
            context_right = blink_example[match.end() :]
            as_blink_format.append(
                {
                    "link": match.group(1),
                    "query_left": context_left.replace("[ENT]", "").replace(
                        "[/ENT]", ""
                    ),
                    "query_right": context_right.replace("[ENT]", "").replace(
                        "[/ENT]", ""
                    ),
                }
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # No need for candidate encoder - just the index
    query_encoder = QueryEncoder.load_model(
        model_path=params["query_encoder_weights_path"]
    )

    index = query_encoder.load_index(params["testing_index_path"])
    query_encoder = query_encoder.to(device)
    candidates, distances = query_encoder.search(
        query=as_blink_format, top_k=5, index=index, device=device, as_titles=True
    )

    for entry, candidate, distance in zip(as_blink_format, candidates, distances):
        entity = entry["link"]
        print(entity)
        print(candidate)
        print(distance)
        print("----")
