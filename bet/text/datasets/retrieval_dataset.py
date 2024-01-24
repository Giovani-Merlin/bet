"""
# * Custom Dataset/collate to be flexible with different encoder/decoder models tokenization (using from default data processing class outputs)
and using hdf5 file
"""

class RetrievalDataset:
    def __init__(self, query, candidate, auxiliar) -> None:
        self.query = dict(query.items())
        self.candidate = dict(candidate.items())
        self.auxiliar = dict(auxiliar.items())

    def __getitem__(self, index):
        query_input = {key: tensor[index] for key, tensor in self.query.items()}
        auxiliar_input = {key: tensor[index] for key, tensor in self.auxiliar.items()}
        # Recover the candidate from the auxiliar index
        candidate_input = {
            key: tensor[auxiliar_input["candidate_index"]]
            for key, tensor in self.candidate.items()
        }

        return query_input, candidate_input, auxiliar_input

    def __len__(self):
        return self.auxiliar["query_index"].shape[0]
