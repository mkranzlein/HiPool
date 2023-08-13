"""Dataset reader for CuRIAM.

Tokens can have multiple labels. This reader should output a list of
tokens for each document and an accompanying list of multiclass labels.

The labels for each document should be [t, num_classes],
where t is the number of tokens in the document.

"""

import json

from hipool.curiam_categories import ORDERED_CATEGORIES


import torch

from torch.utils.data import Dataset


categories_to_ids = {}
for i, category in enumerate(ORDERED_CATEGORIES):
    categories_to_ids[category] = i


class CuriamDataset(Dataset):
    """Reads a file formatted like CuRIAM's corpus.json.

    https://github.com/mkranzlein/curiam/blob/main/corpus/corpus.json
    """

    def __init__(self, json_file):
        data = self.read_json(json_file)
        self.documents = data["documents"]
        self.labels = data["labels"]

    def read_json(self, json_file: str) -> dict:
        with open(json_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        documents = []
        labels = []

        for document in raw_data:
            document_text = []
            document_labels = []
            for sentence in document["sentences"]:
                for token in sentence["tokens"]:
                    token_category_ids = []
                    if "annotations" in token:
                        for annotation in token["annotations"]:
                            annotation_category = annotation["category"]
                            category_id = categories_to_ids[annotation_category]
                            token_category_ids.append(category_id)
                    # Binary multilabels
                    token_labels = torch.zeros(size=(9,))
                    token_labels[token_category_ids] = 1

                    document_text.append(token["text"])
                    document_labels.append(token_labels)
            documents.append(document_text)
            labels.append(torch.stack(document_labels))
        return {"documents": documents, "labels": labels}

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.labels[idx]
