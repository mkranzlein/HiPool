##############################################################
#
# Bert_Classification.py
# This file contains the code for fine-tuning BERT using a
# simple classification head.
#
##############################################################

from hipool.utils import kronecker_generator
from hipool.Graph_Models import GCN, GAT, GraphSAGE, SimpleRank, LinearFirst, DiffPool, HiPool

import networkx as nx
import numpy as np
import transformers
import torch

from torch import nn
from torch_geometric.utils import from_networkx
from torch.nn.utils.rnn import pad_sequence


class Bert_Classification_Model(nn.Module):
    """ A Model for bert fine tuning """

    def __init__(self):
        super().__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.out = nn.Linear(768, 10)

    def forward(self, ids, mask, token_type_ids):
        """ Define how to perfom each call

        Parameters
        __________
        ids: array
            -
        mask: array
            -
        token_type_ids: array
            -

        Returns
        _______
            -
        """
        'original'
        results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        return self.out(results[1])


class Hi_Bert_Classification_Model_GCN(nn.Module):
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self, args, num_class, device, adj_method, pooling_method='mean'):
        super().__init__()
        self.args = args
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = 2
        'default 128 and 32'
        self.lstm_hidden_size = args.lstm_dim
        self.hidden_dim = args.hid_dim

        # self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method = pooling_method

        self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)

        # Start GCN
        if self.args.graph_type == 'gcn':
            self.gcn = GCN(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'gat':
            self.gcn = GAT(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'graphsage':
            self.gcn = GraphSAGE(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'linear':
            self.gcn = LinearFirst(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'rank':
            self.gcn = SimpleRank(input_dim=self.lstm_hidden_size, hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'diffpool':
            self.gcn = DiffPool(self.device, max_nodes=10, input_dim=self.lstm_hidden_size,
                                hidden_dim=32, output_dim=num_class).to(device)
        elif self.args.graph_type == 'hipool':
            self.gcn = HiPool(self.device, input_dim=self.lstm_hidden_size,
                              hidden_dim=32, output_dim=num_class).to(device)

        self.adj_method = adj_method

    def forward(self, ids, mask, token_type_ids):

        bert_ids = pad_sequence(ids).permute(1, 0, 2).long().to(self.device)
        bert_mask = pad_sequence(mask).permute(1, 0, 2).long().to(self.device)
        bert_token_type_ids = pad_sequence(token_type_ids).permute(1, 0, 2).long().to(self.device)
        batch_bert = []
        for emb_pool, emb_mask, emb_token_type_ids in zip(bert_ids, bert_mask, bert_token_type_ids):
            results = self.bert(emb_pool, attention_mask=emb_mask, token_type_ids=emb_token_type_ids)
            batch_bert.append(results[1])

        sent_bert = torch.stack(batch_bert, 0)

        sent_bert = self.mapping(sent_bert)
        node_number = sent_bert.shape[1]

        if self.adj_method == 'random':
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)
        elif self.adj_method == 'er':
            generated_adj = nx.erdos_renyi_graph(node_number, node_number)
        elif self.adj_method == 'binom':
            generated_adj = nx.binomial_graph(node_number, p=0.5)
        elif self.adj_method == 'path':
            generated_adj = nx.path_graph(node_number)
        elif self.adj_method == 'complete':
            generated_adj = nx.complete_graph(node_number)
        elif self.adj_method == 'kk':
            generated_adj = kronecker_generator(node_number)
        elif self.adj_method == 'watts':
            if node_number-1 > 0:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number-1, p=0.5)
            else:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number, p=0.5)
        elif self.adj_method == 'ba':
            if node_number - 1 > 0:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number-1)
            else:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number)
        elif self.adj_method == 'bigbird':

            # following are attention edges
            attention_adj = np.zeros((node_number, node_number))
            global_attention_step = 2
            attention_adj[:, :global_attention_step] = 1
            attention_adj[:global_attention_step, :] = 1
            np.fill_diagonal(attention_adj, 1)  # fill diagonal with 1
            half_sliding_window_size = 1
            np.fill_diagonal(attention_adj[:, half_sliding_window_size:], 1)
            np.fill_diagonal(attention_adj[half_sliding_window_size:, :], 1)
            generated_adj = nx.from_numpy_matrix(attention_adj)

        else:
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)
        nx_adj = from_networkx(generated_adj)
        adj = nx_adj['edge_index'].to(self.device)

        if self.adj_method == 'complete':
            adj = torch.ones((node_number, node_number)).to_sparse().indices().to(self.device)

        if self.args.graph_type.endswith('pool'):
            # diffpool only accepts dense adj
            adj_matrix = nx.adjacency_matrix(generated_adj).todense()
            adj_matrix = torch.from_numpy(np.asarray(adj_matrix)).to(self.device)
            adj = (adj, adj_matrix)

        # sent_bert shape torch.Size([batch_size, 3, 768])
        gcn_output_batch = []
        for node_feature in sent_bert:

            gcn_output = self.gcn(node_feature, adj)

            # Graph-level read out, summation
            gcn_output = torch.sum(gcn_output, 0)
            gcn_output_batch.append(gcn_output)

        gcn_output_batch = torch.stack(gcn_output_batch, 0)

        # GCN ends

        return gcn_output_batch, generated_adj  # (batch_size, class_number)


class Hi_Bert_Classification_Model_GCN_tokenlevel(nn.Module):
    """ A Model for bert fine tuning, put an lstm on top of BERT encoding """

    def __init__(self, num_class, device, adj_method, pooling_method='mean'):
        super().__init__()
        self.bert_path = 'bert-base-uncased'
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)

        self.lstm_layer_number = 2
        self.lstm_hidden_size = 128
        self.max_len = 1024

        # self.bert_lstm = nn.Linear(768, self.lstm_hidden_size)
        self.device = device
        self.pooling_method = pooling_method

        self.mapping = nn.Linear(768, self.lstm_hidden_size).to(device)

        # start GCN
        # MK: use highpool here, like in the non token level version of this class
        # self.gcn = GCN(input_dim=self.lstm_hidden_size,hidden_dim=32,output_dim=num_class).to(device)
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

        if self.adj_method == 'random':
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)
        elif self.adj_method == 'er':
            generated_adj = nx.erdos_renyi_graph(node_number, node_number)
        elif self.adj_method == 'binom':
            generated_adj = nx.binomial_graph(node_number, p=0.5)
        elif self.adj_method == 'path':
            generated_adj = nx.path_graph(node_number)
        elif self.adj_method == 'complete':
            generated_adj = nx.complete_graph(node_number)
        elif self.adj_method == 'kk':
            generated_adj = kronecker_generator(node_number)
        elif self.adj_method == 'watts':
            if node_number-1 > 0:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number-1, p=0.5)
            else:
                generated_adj = nx.watts_strogatz_graph(node_number, k=node_number, p=0.5)
        elif self.adj_method == 'ba':
            if node_number - 1 > 0:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number-1)
            else:
                generated_adj = nx.barabasi_albert_graph(node_number, m=node_number)
        else:
            generated_adj = nx.dense_gnm_random_graph(node_number, node_number)

        nx_adj = from_networkx(generated_adj)
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

        return gcn_output_batch, generated_adj  # (batch_size, class_number)
