"""Sequence- and Token-level classification models.

b: batch_size
k: longest sequence length (in number of chunks)
c: chunk length (in tokens)

"""

import networkx as nx
import transformers
import torch
from jaxtyping import Float, Integer, jaxtyped
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from typeguard import typechecked

from hipool.hipool import HiPool


class TokenClassificationModel(nn.Module):
    """A chunk-based sequence classification model using HiPool.

    Pooled BERT embeddings (one pooled embedding per chunk) are passed into a
    linear layer and then through HiPool, which uses a graph convolutional
    network and an attention mechanism to capture the relations among the
    chunks.

    The output of the model is a binary prediction about the sequence, such as
    whether the movie review is positive or negative.

    """

    def __init__(self, args, num_labels, chunk_len, device, pooling_method='mean'):
        super().__init__()
        self.args = args
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.bert.requires_grad_(True)
        self.chunk_len = chunk_len + 2
        self.linear_dim = 128
        self.hidden_dim = 64

        self.device = device
        self.pooling_method = pooling_method

        self.gcn_output_dim = 64

        self.linear = nn.Linear(768, self.linear_dim).to(device)
        self.linear2 = nn.Linear(768, num_labels).to(device)

        self.gcn = HiPool(self.device, input_dim=self.linear_dim,
                          hidden_dim=32, output_dim=self.gcn_output_dim).to(device)

    @jaxtyped
    @typechecked
    def forward(self, ids: list[Integer[Tensor, "_ d"]],
                mask: list[Integer[Tensor, "_ d"]],
                token_type_ids: list[Integer[Tensor, "_ d"]]):
        """Forward pass through the HiPool model.

        b: batch_size
        s: longest sequence length (in number of chunks)
        c: chunk length (in tokens)

        The three input lists of tensors get padded to the length of the longest
        list of token IDs.

        Args:
            ids: A list of varied-length tensors of token IDs.
            mask: A list of varied-length tensors of attention masks. All 1s.
            token_type_ids: A list of varied-length tensors token_type_ids.
              All 0s.
        """
        
        # Get hipool embedding
        
        # Forward pass happens on one or more documents
        # One is the minimum because hipool needs all of the document's chunks
        # Pipeline: send document through bert sentence by sentence
        
        # Chunking approaches: equal number of sentences, equal number of tokens,
        #   unequal number of sentences that approximates an equal number of tokens
        

        # Pad such that each sequence has the same number of chunks
        # Padding chunks c-dim vectors, where all the input ids are 0, which is
        # BERT's padding token
        padded_ids: Integer[Tensor, "k b c"] = pad_sequence(ids)
        padded_ids: Integer[Tensor, "b k c"] = padded_ids.permute(1, 0, 2).to(self.device)
        padded_masks: Integer[Tensor, "k b c"] = pad_sequence(mask)
        padded_masks: Integer[Tensor, "b k c"] = padded_masks.permute(1, 0, 2).to(self.device)
        padded_token_type_ids: Integer[Tensor, "k b c"] = pad_sequence(token_type_ids)
        padded_token_type_ids: Integer[Tensor, "b k c"] = padded_token_type_ids.permute(1, 0, 2).to(self.device)
        batch_chunk_embeddings = []
        batch_token_embeddings = []
        for ids, mask, token_type_ids in zip(padded_ids, padded_masks, padded_token_type_ids):
            results = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
            # One 768-dim embedding for each chunk
            pooler_output: Float[Tensor, "k 768"] = results["pooler_output"]
            last_hidden_state: Float[Tensor, "k c 768"] = results["last_hidden_state"]
            batch_token_embeddings.append(last_hidden_state)
            batch_chunk_embeddings.append(pooler_output)

        batch_chunk_embeddings: Float[Tensor, "b k 768"] = torch.stack(batch_chunk_embeddings, 0)

        linear_layer_output: Float[Tensor, "b k lin_dim"] = self.linear(batch_chunk_embeddings)

        num_nodes = linear_layer_output.shape[1]
        graph = nx.path_graph(num_nodes)
        adjacency_matrix = nx.adjacency_matrix(graph).todense()
        adjacency_matrix = torch.from_numpy(adjacency_matrix).to(self.device).float()

        # Pass each sequence through HiPool GCN individually then stack
        gcn_output_batch = []
        for node in linear_layer_output:
            gcn_output = self.gcn(node, adjacency_matrix)
            gcn_output_batch.append(gcn_output)
        gcn_output_batch: Float[Tensor, "b gcn"] = torch.stack(gcn_output_batch)

        batch_token_embeddings: Float[Tensor, "b k c 768"] = torch.stack(batch_token_embeddings, 0)

        batch_model_output = []
        for sequence_token_embedding, sequence_gcn_output in zip(batch_token_embeddings, gcn_output_batch):
            sequence_outputs = []
            sequence_gcn_output = sequence_gcn_output.unsqueeze(0)
            repeated_sequence_gcn_output: Float[Tensor, "c gcn"] = sequence_gcn_output.repeat(self.chunk_len, 1)
            for chunk_token_embedding in sequence_token_embedding:
                # combined_embedding: Float[Tensor, "c 768+gcn"] = torch.cat((chunk_token_embedding,
                #                                                             repeated_sequence_gcn_output), dim=1)
                combined_embedding: Float[Tensor, "c 768"] = chunk_token_embedding
                l2_output: Float[Tensor, "c num_labels"] = self.linear2(combined_embedding)
                #l2_output = nn.functional.sigmoid(l2_output)
                sequence_outputs.append(l2_output)
            sequence_outputs: Float[Tensor, "k c num_labels"] = torch.stack(sequence_outputs)
            batch_model_output.append(sequence_outputs)
        batch_model_output = torch.stack(batch_model_output)
        return batch_model_output
