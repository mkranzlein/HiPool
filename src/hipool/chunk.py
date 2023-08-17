"""Chunking functionality described Section 2 of the HiPool paper.

https://aclanthology.org/2023.acl-short.16/

"""

import math

import torch

from transformers import BertTokenizer


def chunk_document(document: str, labels: torch.Tensor, num_labels: int,
                   token_level: bool, chunk_len: int, overlap_len: int,
                   tokenizer: BertTokenizer) -> dict:
    """Tokenizes a document then splits into chunks of tokens.

    Args:
        num_labels: 1 for simple binary classification. > 1 for multilabel
          classification. Each label is assumed to be binary.
        token_level: If true, each token has a label. Otherwise, each chunk gets
          a label. In the evaluation script, only the predicted label for the
          first chunk is evaluated, functionally making chunk-level labeling
          sequence-level labeling.
    """

    batch_encoding = tokenizer.encode_plus(
        document, pad_to_max_length=False, add_special_tokens=True,
        return_attention_mask=True, return_token_type_ids=True,
        return_tensors="pt")

    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    processed_labels = []

    doc_input_ids = batch_encoding["input_ids"].squeeze()

    start_token = torch.tensor([101], dtype=torch.long)
    end_token = torch.tensor([102], dtype=torch.long)

    num_tokens = len(doc_input_ids) - 2  # Exclude head 101, tail 102
    stride = overlap_len - 2
    num_chunks = math.floor(num_tokens / stride)

    # For token-level classification, each token needs a label.
    # For sequence-level classification, each chunk needs a label.

    mask_list = torch.ones(chunk_len, dtype=torch.long)
    type_list = torch.zeros(chunk_len, dtype=torch.long)
    for chunk_id in range(num_chunks - 1):
        chunk_input_ids = doc_input_ids[0:18]
        chunk_input_ids = doc_input_ids[chunk_id * stride:chunk_id * stride + chunk_len - 2]
        chunk_input_ids = torch.cat((start_token, chunk_input_ids, end_token))
        if token_level:
            zeroed_labels = torch.zeros(num_labels)
            chunk_labels = labels[chunk_id * stride:chunk_id * stride + chunk_len - 2, :]
            # Add 0-labels for start and end tokens
            chunk_labels = torch.cat((zeroed_labels.unsqueeze(dim=0), chunk_labels, zeroed_labels.unsqueeze(dim=0)))
        else:
            # Each chunk gets a label instead of each token.
            chunk_labels = labels

        # MK: in eval, make sure to only eval each token once and don't include start and end toks
        input_ids_list.append(chunk_input_ids)
        attention_mask_list.append(mask_list)
        token_type_ids_list.append(type_list)
        processed_labels.append(chunk_labels)

    if len(input_ids_list) == 0:
        raise ValueError("input_ids_list cannot have length 0")

    return ({
        "ids": input_ids_list,
        "mask": attention_mask_list,
        "token_type_ids": token_type_ids_list,
        "targets": processed_labels,
        "len": [torch.tensor(len(processed_labels), dtype=torch.long)]
    })