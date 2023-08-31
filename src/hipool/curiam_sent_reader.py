"""Dataset reader for CuRIAM.

Tokens can have multiple labels. This reader should output a list of
tokens for each document and an accompanying list of multiclass labels.

The labels for each document should be [t, num_classes],
where t is the number of tokens in the document.

"""

import json
from itertools import chain

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from hipool.chunk import chunk_document
from hipool.curiam_categories import SINGLE_CATEGORY

categories_to_ids = {}
for i, category in enumerate(SINGLE_CATEGORY):
    categories_to_ids[category] = i


class CuriamSentDataset(Dataset):
    """Reads a file formatted like CuRIAM's corpus.json.

    https://github.com/mkranzlein/curiam/blob/main/corpus/corpus.json
    """

    def __init__(self, json_file_path: str, tokenizer: BertTokenizerFast,
                 num_labels, chunk_len: int, overlap_len: int):
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len
        self.documents = self.read_json(json_file_path)

    def read_json(self, json_file_path: str) -> list:
        """Processes CuRIAM dataset json into list of documents.

        Documents are represented as a dictionary, with:

        wordpiece_input_ids: A list of all of the input_ids for the document
        first_subword_mask: A list with 1s indicating a wordpiece is the first
          subword of a token and 0s indicating a wordpiece that is not the first
          subword of a token. This is used for evaluation, since we should only
          calculate metrics based on one subword for each token. Here, we choose
          to use the first.
        labels: Labels for the actual tokens in the document, not the
          wordpieces. Because these are for actual tokens, the dimensions won't
          match the length of `wordpiece_input_ids`. We use the
          `first_subword_mask` later to extract the predictions for just the
          first subwords. The number of first subwords will equal the number of
          tokens.
        """
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        documents = []

        for raw_document in raw_data:
            document_labels = []

            # Get wordpieces and first_subword_mask
            sentences = [[token["text"].lower() for token in sentence["tokens"]]
                         for sentence in raw_document["sentences"][:50]]
            assert max([len(s) for s in sentences]) < 512
            tokenizer_output = self.tokenizer(sentences,
                                              is_split_into_words=True,
                                              return_attention_mask=False,
                                              return_token_type_ids=False,
                                              add_special_tokens=False)
            wordpiece_input_ids = list(chain(*tokenizer_output["input_ids"]))
            first_subword_mask = [get_first_subword_mask(tokenizer_output.word_ids(i)) for i in range(len(sentences))]
            first_subword_mask = list(chain(*first_subword_mask))

            # Get labels for actual tokens
            for sentence in raw_document["sentences"][:50]:
                for token in sentence["tokens"]:
                    token_category_ids = []
                    if "annotations" in token:
                        for annotation in token["annotations"]:
                            annotation_category = annotation["category"]
                            if annotation_category in SINGLE_CATEGORY:
                                category_id = categories_to_ids[annotation_category]
                                token_category_ids.append(category_id)
                    # Binary multilabels
                    token_labels = torch.zeros(self.num_labels, dtype=torch.long)
                    token_labels[token_category_ids] = 1
                    document_labels.append(token_labels)
            document_labels = torch.stack(document_labels)
            num_positive = sum(document_labels)
            documents.append({"wordpiece_input_ids": wordpiece_input_ids,
                              "first_subword_mask": first_subword_mask,
                              "labels": document_labels})
        return documents

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

        chunked_document = chunk_document(document["wordpiece_input_ids"],
                                          document["first_subword_mask"],
                                          chunk_len=self.chunk_len,
                                          overlap_len=self.overlap_len)

        chunked_document["labels"] = document["labels"]
        return chunked_document


# TODO: move to utils?
def get_first_subword_mask(sentence_word_ids: list[int]):
    first_subword_mask = []
    current_word = None
    for word_id in sentence_word_ids:
        if word_id != current_word:
            current_word = word_id
            first_subword_mask.append(1)
        else:
            first_subword_mask.append(0)
    return first_subword_mask
