"""Chunking functionality described Section 2 of the HiPool paper.

https://aclanthology.org/2023.acl-short.16/

k: number of chunks in document
c: chunk length (in tokens)

"""

from itertools import chain

import torch
from jaxtyping import Float, Integer, jaxtyped
from torch import Tensor
from typeguard import typechecked


@jaxtyped
@typechecked
def chunk_document(input_ids: list[int], first_subword_mask, chunk_len: int, overlap_len: int) -> dict:
    """Splits a document into chunks of wordpiece subwords."""

    chunks = {"input_ids": [],
              "first_subword_mask": []}

    current_idx = 0
    while True:
        chunks["input_ids"].append(torch.tensor(input_ids[current_idx: current_idx + chunk_len],
                                                dtype=torch.long))
        if current_idx == 0:
            chunks["first_subword_mask"].append(first_subword_mask[current_idx: current_idx + chunk_len])
        else:
            chunks["first_subword_mask"].append(first_subword_mask[current_idx + overlap_len: current_idx + chunk_len])

        if current_idx + chunk_len >= len(input_ids):
            break
        else:
            current_idx += chunk_len - overlap_len

    last_chunk_len = chunks["input_ids"][-1].shape[0]
    last_chunk_ids_padding = torch.zeros((chunk_len - last_chunk_len), dtype=torch.long)  # 0 is pad token
    chunks["input_ids"][-1] = torch.cat((chunks["input_ids"][-1], last_chunk_ids_padding))

    chunks["input_ids"] = torch.stack(chunks["input_ids"])
    num_chunks = chunks["input_ids"].shape[0]
    cls_tokens = torch.tensor(101, dtype=torch.long).repeat(num_chunks).unsqueeze(dim=1)
    sep_tokens = torch.tensor(102, dtype=torch.long).repeat(num_chunks).unsqueeze(dim=1)
    # Jaxtyping not enforcing dictionary assignments
    chunks["input_ids"]: Integer[Tensor, "k c"] = torch.cat((cls_tokens, chunks["input_ids"], sep_tokens), dim=1)
    chunks["attention_mask"] = chunks["input_ids"].not_equal(0).long()
    chunks["token_type_ids"]: Float[Tensor, "k c"] = torch.zeros(chunks["input_ids"].shape, dtype=torch.long)

    chunks["first_subword_mask"] = list(chain(*chunks["first_subword_mask"]))
    return chunks
