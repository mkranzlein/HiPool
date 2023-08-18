"""Implementation of Hierarchical Pooling (HiPool) for long documents.

Paper: https://aclanthology.org/2023.acl-short.16.pdf
"""

import math

import torch
import torch.nn.functional as F

from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch_geometric.nn import DenseGCNConv
from typeguard import typechecked


class HiPool(torch.nn.Module):
    def __init__(self, device, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.device = device

        self.num_mid_nodes = 10
        self.attention_low_mid = torch.nn.Parameter(torch.zeros(size=(input_dim, input_dim))).to(self.device)
        torch.nn.init.xavier_normal_(self.attention_low_mid.data, gain=1.414)
        self.conv1 = DenseGCNConv(input_dim, hidden_dim)

        self.num_high_level_nodes = math.ceil(self.num_mid_nodes / 2)
        self.attention_mid_high = torch.nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim))).to(self.device)
        torch.nn.init.xavier_normal_(self.attention_mid_high.data, gain=1.414)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)

    @jaxtyped
    @typechecked
    def map_low_to_high(self, num_low_nodes: int,
                        num_high_nodes: int) -> Float[Tensor, "num_low num_high"]:
        """Returns a matrix specifying edges from low to high nodes.

        Each low node gets one edge to a high node. Each high node has an equal
        number of connections to low nodes (to the extent possible).

        This mapping is adjacency matrix A_self from the paper.
        """
        edges_per_high = math.ceil(num_low_nodes / num_high_nodes)
        mapping = torch.eye(num_high_nodes, dtype=torch.float, device=self.device)
        mapping = torch.repeat_interleave(mapping, edges_per_high, dim=0)
        mapping = mapping[:num_low_nodes]
        return mapping

    @jaxtyped
    @typechecked
    def cluster_attention(self, x: Float[Tensor, "s linear"], low_to_high_mapping):
        mid_level_representations = torch.matmul(low_to_high_mapping.t(), x)

        # Intra-cluster attention
        scores = torch.matmul(torch.matmul(mid_level_representations, self.attention_low_mid), x.t())

        # Inter-cluster attention
        inverse_mapping: Float[Tensor, "s mid"] = torch.ones_like(low_to_high_mapping) - low_to_high_mapping
        scores = scores * inverse_mapping.t()
        scores = F.softmax(scores, dim=1)

        output = torch.matmul(scores, x) + mid_level_representations
        return output

    @jaxtyped
    @typechecked
    def forward(self, x: Float[Tensor, "s linear"],
                adjacency_matrix: Float[Tensor, "s s"]):
        """A forward pass through the HiPool model.

        HiPool's structure is repeatable, but the paper only uses two layers,
        so that's what is used here.

        HiPool takes as input an h-dimensional representation for each chunk in
        a sequence. These will be called low-level node representations. We
        describe the interaction between the low- and mid-level nodes. The
        logic for the interaction between the mid- and high-level nodes is
        the same.

        The low-level nodes have edges between them (following a path graph),
        And they also get edges to a mid-level node (the first of the two
        HiPool layers). Each low-level node is connected to one mid-level
        node, and the for each mid-level node, the number of connections to
        a low-level node is the same (except for the last mid-level node,
        which just gets the remaining low-level nodes).

        An attention mechanism calculates scores first among the nodes within a
        cluster and then among all the other clusters. The attention-weighted
        scores are added to the representations of the mid-level nodes.

        A weighted adjacency matrix is created to relate the mid-level nodes to
        each other based on how the low-level nodes were related to each other
        (following eq. 2 from the paper). With a path graph, each mid-level node
        ends up with high weight given to itself and its neighbors.
        """

        # ---------------------------- First Layer --------------------------- #
        num_low_nodes = x.shape[0]
        low_to_mid_mapping: Float[Tensor, "s mid"] = self.map_low_to_high(num_low_nodes,
                                                                          self.num_mid_nodes)

        mid_adjacency_matrix: Float[Tensor, "mid mid"] = torch.matmul(
            torch.matmul(low_to_mid_mapping.t(), adjacency_matrix),
            low_to_mid_mapping)

        mid_representation = self.cluster_attention(x, low_to_mid_mapping)

        # x_mid will be the input to the next layer
        x_mid = self.conv1(mid_representation, mid_adjacency_matrix)
        x_mid = F.relu(x_mid)
        x_mid = F.dropout(x_mid, training=self.training)[0]

        # --------------------------- Second Layer --------------------------- #
        mid_to_high_mapping = self.map_low_to_high(self.num_mid_nodes, self.num_high_level_nodes)

        high_level_adjacency_matrix = torch.matmul(torch.matmul(mid_to_high_mapping.t(), mid_adjacency_matrix),
                                                   mid_to_high_mapping)
        high_representation = self.cluster_attention(x_mid, mid_to_high_mapping)

        # DenseGCNConv's forward() adds a batch dimension that we don't need here
        output = self.conv2(high_representation, high_level_adjacency_matrix).squeeze()

        output = F.relu(output)
        # This was originally output.mean(), but the paper seems to suggest sum gave better performance?
        output = output.mean(dim=0)
        return output
