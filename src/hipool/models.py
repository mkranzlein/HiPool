"""Sequence- and Token-level classification models.

b: batch_size
k: longest sequence length (in number of chunks)
c: chunk length (in tokens)
"""

import networkx as nx
import torch
from jaxtyping import Float, Integer, jaxtyped
from torch import nn, Tensor
from typeguard import typechecked as typechecker

from hipool.hipool import HiPool


class DocModel(nn.Module):
    """Produces a document context embedding via HiPool.

    Pooled BERT embeddings (one pooled embedding per chunk) are passed into a
    linear layer and then through HiPool, which uses a graph convolutional
    network and an attention mechanism to capture the relations among the
    chunks.
    """
    def __init__(self, chunk_len, device, linear_dim=64, hidden_dim=32, output_dim=32):
        super().__init__()
        self.device = device
        self.chunk_len = chunk_len + 2  # Accomodate [CLS] and [SEP]
        self.linear_dim = linear_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(768, self.linear_dim).to(device)

        self.gcn = HiPool(self.device, input_dim=self.linear_dim,
                          hidden_dim=32, output_dim=self.output_dim).to(device)

    def forward(self, chunk_bert_embeddings: dict):
        """Forward pass through HiPool.

        Pooled BERT output for each chunk goes through a linear layer and then
        through HiPool graph convolutional network.

        """
        linear_layer_output: Float[Tensor, "k lin_dim"] = self.linear(chunk_bert_embeddings)

        num_nodes = linear_layer_output.shape[0]  # TODO: might need different index
        graph = nx.path_graph(num_nodes)
        adjacency_matrix = nx.adjacency_matrix(graph).todense()
        adjacency_matrix = torch.from_numpy(adjacency_matrix).to(self.device).float()

        doc_hipool_embedding = self.gcn(linear_layer_output, adjacency_matrix)
        return doc_hipool_embedding


class SentenceClassificationModel(nn.Module):
    """Sentence classification model via BERT.

    Predicts whether sentence has any metalinguistic tokens.
    """
    def __init__(self, num_labels, bert_model, device):
        super().__init__()
        self.bert = bert_model
        self.device = device
        self.linear = nn.Linear(768, num_labels).to(device)

    @jaxtyped(typechecker=typechecker)
    def forward(self, ids: Integer[Tensor, "_ c"],
                mask: Integer[Tensor, "_ c"],
                token_type_ids: Integer[Tensor, "_ c"]):
        """Forward pass."""

        # last_hidden_state is x[0], pooler_output is x[1]
        x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)["pooler_output"]
        output = self.linear(x)
        return output


class TokenClassificationModel(nn.Module):
    """Token classification via BERT and optional document embedding."""

    def __init__(self, num_labels, bert_model, device, use_doc_embedding=False, doc_embedding_dim=None):
        super().__init__()
        self.bert = bert_model
        self.device = device
        self.use_doc_embedding = use_doc_embedding
        self.doc_embedding_dim = doc_embedding_dim
        if self.use_doc_embedding:
            self.linear = nn.Linear(768 + self.doc_embedding_dim, num_labels).to(device)
        else:
            self.linear = nn.Linear(768, num_labels).to(device)

    @jaxtyped(typechecker=typechecker)
    def forward(self, ids: Integer[Tensor, "_ c"],
                mask: Integer[Tensor, "_ c"],
                token_type_ids: Integer[Tensor, "_ c"],
                doc_embedding=None):
        """Forward pass."""

        # last_hidden_state is x[0], pooled_output is x[1]
        x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)["last_hidden_state"]
        if self.use_doc_embedding:
            repeated_doc_embedding = doc_embedding.repeat(x.shape[0], x.shape[1], 1)
            x1 = torch.cat((x, repeated_doc_embedding), dim=2)
            output = self.linear(x1)
        else:
            output = self.linear(x)
        return output
