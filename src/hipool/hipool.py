from math import ceil

import torch
import torch.nn.functional as F

from torch_geometric.nn import DenseGCNConv


# Following two methods are our hi-method: Feb, 2022
class HiPool(torch.nn.Module):
    def __init__(self, device, input_dim, hidden_dim, output_dim):
        super().__init__()  # hid dim 32

        self.device = device
        self.num_nodes1 = 10
        self.num_nodes2 = ceil(self.num_nodes1 / 2)

        self.conv1 = DenseGCNConv(input_dim, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)

        # output layer
        self.linear1 = torch.nn.Linear(hidden_dim, output_dim)

        # cross-layer attention, l1
        self.cross_attention_l1 = torch.nn.Parameter(torch.zeros(size=(input_dim, input_dim))).to(self.device)
        torch.nn.init.xavier_normal_(self.cross_attention_l1.data, gain=1.414)

        # cross-layer attention, l2
        self.cross_attention_l2 = torch.nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim))).to(self.device)
        torch.nn.init.xavier_normal_(self.cross_attention_l2.data, gain=1.414)

        # reversed linear layer, l1
        self.reversed_l1 = torch.nn.Parameter(torch.zeros(size=(hidden_dim, input_dim))).to(self.device)
        torch.nn.init.xavier_normal_(self.reversed_l1.data, gain=1.414)

        self.reversed_conv1 = DenseGCNConv(input_dim, hidden_dim)

        # add self-attention for l1
        self.multihead_attn_l1 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=2)

    def forward(self, x, adjacency_matrix):
        # forward_cross_best

        # hipool: add sent-token cross-attention (cross-layer) attention: 2 layers
        portion1 = ceil(x.shape[0] / self.num_nodes1)
        flat_s = torch.eye(self.num_nodes1)  # identity matrix of num_nodes1 x num_nodes1
        flat_s = torch.repeat_interleave(flat_s, portion1, dim=0)[:x.shape[0], ].float().to(self.device)

        # first layer
        x1 = torch.matmul(flat_s.t(), x)  # (5,128)
        self.adj1 = torch.matmul(torch.matmul(flat_s.t(), adjacency_matrix), flat_s)

        # Testing cross-layer attention'
        # generate inverse adj for cross-layer attention
        reverse_s = torch.ones_like(flat_s) - flat_s
        scores = torch.matmul(torch.matmul(x1, self.cross_attention_l1), x.t())
        # mask own cluster and do cross-cluster
        scores = scores * reverse_s.t()
        alpha = F.softmax(scores, dim=1)
        # compute \alpha * x
        x1 = torch.matmul(alpha, x) + x1
        # Cross-layer ends

        x = self.conv1(x1, self.adj1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)[0]

        # Second layer
        portion2 = ceil(x.shape[0] / self.num_nodes2)
        flat_s = torch.eye(self.num_nodes2)
        flat_s = torch.repeat_interleave(flat_s, portion2, dim=0)[:x.shape[0], ].float().to(self.device)

        x2 = torch.matmul(flat_s.t(), x)
        self.adj2 = torch.matmul(torch.matmul(flat_s.t(), self.adj1), flat_s)

        # Testing cross-layer attention for 2nd layer
        # generate inverse adj for cross-layer attention
        reverse_s = torch.ones_like(flat_s) - flat_s
        scores = torch.matmul(torch.matmul(x2, self.cross_attention_l2), x.t())
        # mask own cluster and do cross-cluster
        scores = scores * reverse_s.t()
        alpha = F.softmax(scores, dim=1)
        # compute \alpha * x
        x2 = torch.matmul(alpha, x) + x2
        'cross-layer for 2nd layer ends'

        # DenseGCNConv's forward() adds a batch dimension that we don't need here
        x = self.conv2(x2, self.adj2).squeeze()

        'return mean'
        x = x.mean(dim=0)
        x = F.relu(self.linear1(x))
        # Normalize output and take log for numerical stability
        output = F.log_softmax(x, dim=0)
        return output
