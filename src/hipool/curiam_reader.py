"""Dataset reader for CuRIAM.

Tokens can have multiple labels. This reader should output a list of
tokens for each document and an accompanying list of multiclass labels.

The labels for each document should be [t, num_classes],
where t is the number of tokens in the document.

"""

import json

from torch.utils.data import Dataset


with open("data/curiam_sample.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ORDERED_CATEGORIES = [
    "Focal Term",
    "Definition",
    "Metalinguistic Cue",
    "Direct Quote",
    "Legal Source",
    "Language Source",
    "Named Interpretive Rule",
    "Example Use",
    "Appeal to Meaning",
]

categories_to_ids = {}
for i, category in enumerate(ORDERED_CATEGORIES):
    categories_to_ids[category] = i

documents = []

for document in data:
    document_text = []
    document_labels = []
    for sentence in document["sentences"]:
        for token in sentence["tokens"]:
            token_labels = []
            if "annotations" in token:
                for annotation in token["annotations"]:
                    token_labels.append(annotation["category"])
