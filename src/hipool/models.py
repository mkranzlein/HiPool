##############################################################
#
# Bert_Classification.py
# This file contains the code for fine-tuning BERT using a
# simple classification head.
#
##############################################################

from hipool.hipool import HiPool

import networkx as nx
import transformers
import torch
import torch_geometric

from jaxtyping import Float, Integer, jaxtyped
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from typeguard import typechecked


class ChunkModel(nn.Module):
    """A chunk-based sequence classification model using HiPool.

    Pooled BERT embeddings (one pooled embedding per chunk) are passed into a
    linear layer and then through HiPool, which uses a graph convolutional
    network and an attention mechanism to capture the relations among the
    chunks.

    The output of the model is a binary prediction about the sequence, such as
    whether the movie review is positive or negative.

    """

    def __init__(self, args, num_class, device, pooling_method='mean'):
        super().__init__()
        self.args = args
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")

        self.lstm_hidden_size = 128
        self.hidden_dim = 32

        self.device = device
        self.pooling_method = pooling_method

        self.linear = nn.Linear(768, 128).to(device)

        self.gcn = HiPool(self.device, input_dim=self.lstm_hidden_size,
                          hidden_dim=32, output_dim=num_class).to(device)


    @jaxtyped
    @typechecked
    def forward(self, ids: list[Tensor], mask: list[Tensor],
                token_type_ids: list[Tensor]):
        """Forward pass through the HiPool model.

        b: batch_size
        s: longest sequence length (in number of chunks)
        c: chunk length (in tokens)
        """

        # Pad such that each sequence has the same number of chunks
        # Padding chunks c-dim vectors, where all the input ids are 0, which is
        # BERT's padding token
        padded_ids: Integer[Tensor, "s b c"] = pad_sequence(ids)
        padded_ids: Integer[Tensor, "b s c"] = padded_ids.permute(1, 0, 2).to(self.device)
        padded_masks: Integer[Tensor, "s b c"] = pad_sequence(mask)
        padded_masks: Integer[Tensor, "b s c"] = padded_masks.permute(1, 0, 2).to(self.device)
        padded_token_type_ids: Integer[Tensor, "s b c"] = pad_sequence(token_type_ids)
        padded_token_type_ids: Integer[Tensor, "b s c"] = padded_token_type_ids.permute(1, 0, 2).to(self.device)
        batch_chunk_embeddings = []
        for ids, mask, token_type_ids in zip(padded_ids, padded_masks, padded_token_type_ids):
            results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
            # One 768-dim embedding for each chunk
            pooler_output: Float[Tensor, "s 768"] = results["pooler_output"]
            batch_chunk_embeddings.append(pooler_output)

        batch_chunk_embeddings: Float[Tensor, "b s 768"] = torch.stack(batch_chunk_embeddings, 0)

        linear_layer_output: Float[Tensor, "b s 128"] = self.linear(batch_chunk_embeddings)

        num_nodes = linear_layer_output.shape[1]
        graph = nx.path_graph(num_nodes)
        adjacency_matrix = nx.adjacency_matrix(graph).todense()
        adjacency_matrix = torch.from_numpy(adjacency_matrix).to(self.device).float()

        # Pass each sequence through HiPool GCN individually then stack
        gcn_output_batch = []
        for node in linear_layer_output:

            gcn_output = self.gcn(node, adjacency_matrix)
            gcn_output_batch.append(gcn_output)
        gcn_output_batch = torch.stack(gcn_output_batch)

        return gcn_output_batch


class TokenLevelModel(nn.Module):
    """A token-based sequence classification model using HiPool."""

    def __init__(self, num_class, device, adj_method, pooling_method='mean'):
        super().__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = 2
        self.lstm_hidden_size = 128
        self.max_len = 1024

        self.device = device
        self.pooling_method = pooling_method

        self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)

        # self.gcn = HiPool(self.device, input_dim=self.lstm_hidden_size,
        #                   hidden_dim=32, output_dim=num_class).to(device)
        from hipool.Graph_Models import GAT

        self.gcn = GAT(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        self.adj_method = adj_method

    def forward(self, ids, mask, token_type_ids):
        batch_size = len(ids)
        reshape_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        reshape_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        reshape_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)

        batch_bert = []
        for emb_pool, emb_mask, emb_token_type_ids in zip(reshape_ids, reshape_mask, reshape_token_type_ids):
            results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
            batch_bert.append(results[0])  # results[0] shape: (length,chunk_len, 768)

        sent_bert = torch.stack(batch_bert, 0).reshape(batch_size, -1, 768)[:, :self.max_len, :]

        # GCN starts
        sent_bert = self.mapping(sent_bert)
        node_number = sent_bert.shape[1]

        # random, using networkx

        generated_adj = nx.path_graph(node_number)

        nx_adj = torch_geometric.utils.from_networkx(generated_adj)
        adj = nx_adj['edge_index'].to(self.device)

        if self.adj_method == 'complete':
            # complete connected
            adj = torch.ones((node_number, node_number)).to_sparse().indices().to(self.device)

        # sent_bert shape torch.Size([batch_size, 3, 768])
        gcn_output_batch = []
        for node_feature in sent_bert:
            gcn_output = self.gcn(node_feature, adj)

            'graph-level read out, summation'
            gcn_output = torch.sum(gcn_output, 0)
            gcn_output_batch.append(gcn_output)

        gcn_output_batch = torch.stack(gcn_output_batch, 0)

        # GCN ends

        return gcn_output_batch