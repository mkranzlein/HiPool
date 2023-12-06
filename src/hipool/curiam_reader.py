"""Dataset reader for CuRIAM.

Tokens can have multiple labels. This reader should output a list of
tokens for each document and an accompanying list of multiclass labels.

The labels for each document should be [t, num_classes],
where t is the number of tokens in the document.

"""

import json
from itertools import chain
from typing import List

import torch
import transformers
from jaxtyping import Float, Integer, jaxtyped
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from hipool.chunk import chunk_document
from hipool.curiam_categories import REDUCED_CATEGORIES


class DocDataset(Dataset):
    """Reads a file formatted like CuRIAM's corpus.json.

    https://github.com/mkranzlein/curiam/blob/main/corpus/corpus.json
    """

    def __init__(self, json_file_path: str, tokenizer: BertTokenizerFast,
                 bert_model: transformers.BertModel, num_labels: int,
                 chunk_len: int, overlap_len: int):
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len
        self.documents = self.read_json(json_file_path)
        self.bert = bert_model

    def read_json(self, json_file_path: str) -> list:
        """Processes CuRIAM dataset json into list of documents.

        Documents are represented as a dictionary, with:

        sentences: a list of tokens.
        input_ids: wordpiece input_ids for BERT.
        wordpiece_input_ids: A list of all of the input_ids for the document
          subword_mask: A list with 1s indicating a wordpiece is the first
          subword of a token and 0s indicating a wordpiece that is not the first
          subword of a token. This is used for evaluation, since we should only
          calculate metrics based on one subword for each token. Here, we choose
          to use the first.
        labels: Labels for the actual tokens in the document, not the
          wordpieces. Because these are for actual tokens, the dimensions won't
          match the length of `wordpiece_input_ids`. We use the
          `subword_mask` later to extract the predictions for just the
          first subwords. The number of first subwords will equal the number of
          tokens.
        """
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # Each document is a list of sentences, and each sentence is a list of tokens.
        documents = []

        # labels[i] is an [n, k] tensor where n is the number of tokens in the i-th sentence and
        # k is the number of binary labels assigned to each token.

        for raw_document in json_data:
            doc_sentences = [[token["text"].lower() for token in sentence["tokens"]]
                             for sentence in raw_document["sentences"]]
            doc_labels = [get_multilabel(sentence, REDUCED_CATEGORIES)
                          for sentence in raw_document["sentences"]]

            tokenizer_output = self.tokenizer(doc_sentences,
                                              is_split_into_words=True,
                                              return_attention_mask=False,
                                              return_token_type_ids=False,
                                              add_special_tokens=True)
            wordpiece_input_ids = tokenizer_output["input_ids"]
            subword_mask = [get_subword_mask(tokenizer_output.word_ids(i))
                            for i in range(len(doc_sentences))]
            # subword_mask = list(chain(*subword_mask))

            if len(doc_sentences) > 150 or len(doc_sentences) < 10:
                continue
            documents.append({"sentences": doc_sentences, "input_ids": wordpiece_input_ids,
                              "subword_mask": subword_mask, "labels": doc_labels})
        return documents

    def __len__(self) -> int:
        """Returns the number of documents in the dataset."""
        return len(self.documents)

    @jaxtyped
    def __getitem__(self, idx) -> tuple:
        """Returns one document from the dataset by index.

        This includes the sentences, the labels, and a chunked version of the
        document.

        Used by a dataloader during training.
        """

        # chunked documents should include bert embedding tensors

        # When to tokenize?? Sentences need cls, sep, and pad
        # Pad now with labels?
        document = self.documents[idx]

        doc_input_ids = list(chain(*[sent_ids[1:-1] for sent_ids in document["input_ids"]]))
        doc_input_ids = doc_input_ids
        chunked_document = chunk_document(doc_input_ids,
                                          chunk_len=self.chunk_len,
                                          overlap_len=self.overlap_len)

        chunks_input_ids = chunked_document["input_ids"].cuda()
        chunks_attention_mask = chunked_document["attention_mask"].cuda()
        chunks_token_type_ids = chunked_document["token_type_ids"].cuda()

        chunk_bert_embeddings = []
        for ids, mask, token_type_ids in zip(chunks_input_ids,
                                             chunks_attention_mask,
                                             chunks_token_type_ids):
            chunk_embedding = self.bert(ids.unsqueeze(0), attention_mask=mask.unsqueeze(0),
                                        token_type_ids=token_type_ids.unsqueeze(0))["pooler_output"]
            chunk_bert_embeddings.append(chunk_embedding.squeeze())

        chunk_bert_embeddings: Float[Tensor, "k 768"] = torch.stack(chunk_bert_embeddings, 0)

        sent_dataset = SentDataset(document)
        return sent_dataset, chunk_bert_embeddings


class SentDataset(Dataset):
    def __init__(self, document: dict):
        self.sentences = document["sentences"]
        self.input_ids = document["input_ids"]
        self.subword_mask = document["subword_mask"]
        self.labels = document["labels"]

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"sentence": self.sentences[idx], "input_ids": self.input_ids[idx],
                "subword_mask": self.subword_mask[idx], "labels": self.labels[idx]}


def get_multilabel(sentence: List[dict], applicable_categories: list) -> Integer[Tensor, "n k"]:
    """Returns labels for binary multilabel classification for all tokens in a sentence.

    For example, if the two classes are direct quote and definition,

    A token would have the label:
    - [1, 1] if part of a direct quote and a defintion
    - [1, 0] if part of a direct quote but not a definition
    """
    categories_to_ids = {}
    for i, category in enumerate(applicable_categories):
        categories_to_ids[category] = i
    token_category_ids = []

    labels = []
    for token in sentence["tokens"]:
        if "annotations" in token:
            for annotation in token["annotations"]:
                annotation_category = annotation["category"]
                if annotation_category in applicable_categories:
                    category_id = categories_to_ids[annotation_category]
                    token_category_ids.append(category_id)
        # Binary multilabels
        token_label = torch.zeros(len(applicable_categories), dtype=torch.long)
        token_label[token_category_ids] = 1
        labels.append(token_label)
    labels = torch.stack(labels)
    return labels


# TODO: move to utils?
def get_subword_mask(sentence_word_ids: list[int]):
    """Returns mask indicating whether subwords are first in a token.

    1 if subword is first part of token else 0.

    """
    subword_mask = []
    current_word = None
    for word_id in sentence_word_ids:
        # Ignore special tokens [CLS] and [SEP] which have word_id=None
        if word_id is None:
            continue
        if word_id != current_word:
            current_word = word_id
            subword_mask.append(1)
        else:
            subword_mask.append(0)
    return subword_mask
