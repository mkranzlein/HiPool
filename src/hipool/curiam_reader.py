"""Dataset reader for CuRIAM.

Tokens can have multiple labels. This reader should output a list of
tokens for each document and an accompanying list of multiclass labels.

The labels for each document should be [t, num_classes],
where t is the number of tokens in the document.

"""

import json

from hipool.curiam_categories import ORDERED_CATEGORIES
from hipool.chunk import chunk_document
import torch

from torch.utils.data import Dataset

categories_to_ids = {}
for i, category in enumerate(ORDERED_CATEGORIES):
    categories_to_ids[category] = i


class CuriamDataset(Dataset):
    """Reads a file formatted like CuRIAM's corpus.json.

    https://github.com/mkranzlein/curiam/blob/main/corpus/corpus.json
    """

    def __init__(self, json_file_path, tokenizer, max_len, chunk_len, overlap_len):
        processed_json = self.read_json(json_file_path)
        self.documents = processed_json["documents"]
        self.labels = processed_json["labels"]
        self.num_class = 9
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len

    def read_json(self, json_file_path: str) -> dict:
        with open(json_file_path, "r", encoding="utf-8") as f:
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

    def shuffle(self, seed) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx) -> dict:
        """Returns a specified preprocessed document from the dataset along with
        its labels.

        Used by the dataloaders during training.
        """

        document = self.documents[idx]
        labels = self.labels[idx]

        chunked_document = chunk_document(document, labels,
                                          num_labels=9, token_level=True,
                                          chunk_len=20, overlap_len=10,
                                          tokenizer=self.tokenizer)

        return chunked_document
