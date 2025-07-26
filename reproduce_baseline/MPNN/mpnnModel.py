import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set, global_add_pool
from torch_geometric.data import Data, Batch

from typing import List, Optional, Union, Tuple, Dict


class CustomMPNN(nn.Module):
    def __init__(self,
                 node_in_feats: int,
                 edge_in_feats: int,
                 node_out_feats: int,
                 edge_hidden_feats: int,
                 num_step_message_passing: int = 3,
                 residual: bool = True,
                 aggregator_type: str = 'add'):
        super(CustomMPNN, self).__init__()

        self.node_in_feats = node_in_feats
        self.node_out_feats = node_out_feats
        self.num_step_message_passing = num_step_message_passing
        self.residual = residual

        # Edge Network (like in DGL NNConv)
        self.edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )

        self.gnn_layer = NNConv(in_channels=node_out_feats,
                                out_channels=node_out_feats,
                                nn=self.edge_network,
                                aggr=aggregator_type)

        # Project initial node features to node_out_feats dim
        self.node_proj = nn.Linear(node_in_feats, node_out_feats)

    def forward(self, x, edge_index, edge_attr):
        h = self.node_proj(x)

        for _ in range(self.num_step_message_passing):
            h_in = h
            h = self.gnn_layer(h, edge_index, edge_attr)
            if self.residual:
                h = h + h_in
        return h


class CustomFFN(nn.Module):
    def __init__(self, d_input: int, d_hidden_list: List[int], d_output: int,
                 activation: str = 'relu', dropout_p: float = 0.0,
                 dropout_at_input_no_act: bool = True):
        super(CustomFFN, self).__init__()

        layers = []
        if dropout_at_input_no_act and dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        prev_dim = d_input
        for hidden_dim in d_hidden_list:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim

        self.ffn_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, d_output)

    def forward(self, x):
        embeddings = self.ffn_layers(x)
        out = self.output_layer(embeddings)
        return embeddings, out


class MPNNPOM_PyG(nn.Module):
    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = 'add',
                 mode: str = 'classification',
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 readout_type: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True):

        super(MPNNPOM_PyG, self).__init__()

        self.mode = mode
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.readout_type = readout_type

        if mode == 'classification':
            self.ffn_output = n_tasks * n_classes
        else:
            self.ffn_output = n_tasks

        self.mpnn = CustomMPNN(node_in_feats=number_atom_features,
                               edge_in_feats=number_bond_features,
                               node_out_feats=node_out_feats,
                               edge_hidden_feats=edge_hidden_feats,
                               num_step_message_passing=num_step_message_passing,
                               residual=mpnn_residual,
                               aggregator_type=message_aggregator_type)

        self.project_edge_feats = nn.Sequential(
            nn.Linear(number_bond_features, edge_out_feats),
            nn.ReLU()
        )

        total_feat_dim = node_out_feats + edge_out_feats

        if readout_type == 'set2set':
            self.readout = Set2Set(total_feat_dim,
                                   processing_steps=num_step_set2set,
                                   num_layers=num_layer_set2set)
            ffn_input_dim = 2 * total_feat_dim
        elif readout_type == 'global_sum_pooling':
            self.readout = global_add_pool
            ffn_input_dim = total_feat_dim
        else:
            raise ValueError(f"Unsupported readout: {readout_type}")

        d_hidden_list = ffn_hidden_list + [ffn_embeddings]
        self.ffn = CustomFFN(ffn_input_dim, d_hidden_list, self.ffn_output,
                             activation=ffn_activation,
                             dropout_p=ffn_dropout_p,
                             dropout_at_input_no_act=ffn_dropout_at_input_no_act)

    def forward(self, data: Batch):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        node_emb = self.mpnn(x, edge_index, edge_attr)
        edge_emb = self.project_edge_feats(edge_attr)

        # Prepare node-level messages with edge features (radius 0 fusion)
        row, col = edge_index
        edge_msg = torch.zeros_like(node_emb)
        edge_msg = edge_msg.index_add(0, row, edge_emb)

        node_fused = torch.cat([node_emb, edge_msg], dim=1)
        
        if self.readout_type == 'set2set':
            graph_emb = self.readout(node_fused, batch)
        else:
            graph_emb = self.readout(node_fused, batch)

        embeddings, out = self.ffn(graph_emb)

        if self.mode == 'classification':
            logits = out.view(-1, self.n_tasks, self.n_classes)
            proba = torch.sigmoid(logits)
            if self.n_classes == 1:
                proba = proba.squeeze(-1)
            return proba, logits, embeddings
        else:
            return out
