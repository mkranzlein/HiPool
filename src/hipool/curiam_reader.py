"""Dataset reader for CuRIAM.

Tokens can have multiple labels. This reader should output a list of
tokens for each document and an accompanying list of multiclass labels.

The labels for each document should be [t, num_classes],
where t is the number of tokens in the document.

"""

import json
import math

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

    def __init__(self, json_file, tokenizer, max_len, chunk_len, overlap_len):
        _ = self.read_json(json_file)
        self.documents = _["documents"]
        self.labels = _["labels"]
        self.num_classes = 9
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len

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

    def shuffle(self, seed) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.documents)

    def chunk_document(self, document, targets):
        document = " ".join(document)

        batch_encoding = self.tokenizer.encode_plus(
            document,
            max_length=self.max_len,
            pad_to_max_length=False,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_tensors='pt')

        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = batch_encoding["input_ids"].squeeze()

        start_token = torch.tensor([101], dtype=torch.long)
        end_token = torch.tensor([102], dtype=torch.long)

        total_token = len(previous_input_ids) - 2  # Exclude head 101, tail 102
        stride = self.overlap_len - 2
        number_chunks = math.floor(total_token/stride)

        mask_list = torch.ones(self.chunk_len, dtype=torch.long)
        type_list = torch.zeros(self.chunk_len, dtype=torch.long)
        zeroed_labels = torch.zeros(self.num_classes)
        for current in range(number_chunks-1):
            input_ids = previous_input_ids[current*stride:current*stride+self.chunk_len-2]
            chunk_targets = targets[current*stride:current*stride+self.chunk_len-2,:]
            input_ids = torch.cat((start_token, input_ids, end_token))
            # MK: in eval, make sure to only eval each token once and don't include start and end toks
            chunk_targets = torch.cat((zeroed_labels.unsqueeze(dim=0), chunk_targets, zeroed_labels.unsqueeze(dim=0)))
            input_ids_list.append(input_ids)

            attention_mask_list.append(mask_list)
            token_type_ids_list.append(type_list)
            targets_list.append(chunk_targets)

        if len(input_ids_list) == 0:
            raise ValueError("input_ids_list cannot have length 0")

        return ({
            'ids': input_ids_list,
            'mask': attention_mask_list,
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        })

    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""

        document = self.documents[idx]
        targets = self.labels[idx]

        chunked_document = self.chunk_document(document, targets)

        return chunked_document
