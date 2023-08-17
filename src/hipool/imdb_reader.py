import math
import re

import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from hipool.chunk import chunk_document


class IMDBDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len, chunk_len,
                 overlap_len):
        processed_data = self.read_data(file_path)
        self.documents = processed_data["documents"]
        self.labels = processed_data["labels"]
        self.num_labels = 1
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len

    def read_data(self, file_path):
        df = pd.read_csv(file_path)
        df = df[df.review.notnull()]
        df = df.rename(columns={"review": "text",
                                "sentiment": "label"})

        LE = LabelEncoder()
        # Turns "positive" and "negative" text labels into 1 and 0
        df["label"] = LE.fit_transform(df["label"])
        df['text'] = df.text.apply(self.clean_txt)
        documents = list(df["text"])
        labels = torch.tensor(df["label"])
        return {"documents": documents, "labels": labels}

    def clean_txt(self, text):
        """ Removes special characters from text.

        Not sure how necessary this is for this dataset.
        """

        text = re.sub("'", "", text)
        text = re.sub("(\\W)+", " ", text)
        return text

    def chunk_document(self, data_tokenize, targets):

        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []
        previous_input_ids = data_tokenize["input_ids"].squeeze()
        targets = torch.tensor(targets, dtype=torch.int)

        start_token = torch.tensor([101], dtype=torch.long)
        end_token = torch.tensor([102], dtype=torch.long)

        total_token = len(previous_input_ids) - 2  # remove head 101, tail 102
        stride = self.overlap_len - 2
        number_chunks = math.floor(total_token / stride)

        mask_list = torch.ones(self.chunk_len, dtype=torch.long)
        type_list = torch.zeros(self.chunk_len, dtype=torch.long)
        for current in range(number_chunks - 1):
            input_ids = previous_input_ids[current * stride:current * stride + self.chunk_len - 2]
            input_ids = torch.cat((start_token, input_ids, end_token))
            input_ids_list.append(input_ids)

            attention_mask_list.append(mask_list)
            token_type_ids_list.append(type_list)
            targets_list.append(targets)

        if len(input_ids_list) == 0:
            input_ids = torch.ones(self.chunk_len - 2, dtype=torch.long)
            input_ids = torch.cat((start_token, input_ids, end_token))
            input_ids_list.append(input_ids)

            attention_mask_list.append(mask_list)
            token_type_ids_list.append(type_list)
            targets_list.append(targets)

        return ({
            'ids': input_ids_list,  # torch.tensor(ids, dtype=torch.long),
            # torch.tensor(mask, dtype=torch.long),
            'mask': attention_mask_list,
            # torch.tensor(token_type_ids, dtype=torch.long),
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        })

    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""

        document = self.documents[idx]
        labels = self.labels[idx].unsqueeze(0)

        chunked_document = chunk_document(document, labels,
                                          num_labels=1, token_level=False,
                                          chunk_len=20, overlap_len=10,
                                          tokenizer=self.tokenizer)
        return chunked_document

    def __len__(self):
        """ Return data length """
        return len(self.documents)
