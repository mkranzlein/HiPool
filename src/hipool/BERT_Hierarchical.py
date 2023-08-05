##############################################################
#
# BERT_Hierarchical.py
# This file contains the code to fine-tune BERT by computing
# segment tensors as a pooled result from all the segments
# obtained after tokenization.
#
##############################################################

from hipool.TransformerLayer import BERT

import torch
import transformers

from torch import nn


class BERT_Hierarchical_BERT_Model(nn.Module):
    def __init__(self, device, pooling_method="mean", lstm_layer_number=1, lstm_hidden_size=32):
        super().__init__()

        self.pooling_method = pooling_method
        self.device = device

        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = lstm_layer_number
        self.lstm_hidden_size = lstm_hidden_size

        self.mapping = nn.Linear(768, lstm_hidden_size)
        self.BERTLayer = BERT(hidden=lstm_hidden_size, n_layers=1, attn_heads=8).to(device)
        self.out = nn.Linear(self.lstm_hidden_size, 10)

    def forward(self, ids, mask, token_type_ids, length):
        # length is a list [2,2,2]

        results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        chunks_emb = results[1].split_with_sizes(length)

        # lstm starts
        max_step = max(length)
        batch_size = len(length)
        lstm_input = torch.zeros(batch_size, max_step, 768).to(self.device)

        for current_id, element in enumerate(lstm_input):
            # todo: deal with different shapes
            lstm_input[current_id] = chunks_emb[current_id]

        lstm_input = lstm_input.permute(1, 0, 2)
        # shape: torch.Size([2, 3, 768]) [len, batch_size, dim]

        # lstm ends
        lstm_input = self.mapping(lstm_input)
        lstm_output = self.BERTLayer(lstm_input)
        'outputs.shape torch.Size([2, 3, 64]),emb_pool shape torch.Size([3, 64])'

        return self.out(lstm_output[-1])

# layer num, head num
# 1,1, 0.823220536756126
# 2,1, 0.8165110851808635
# 1,8,  0.8211785297549592
