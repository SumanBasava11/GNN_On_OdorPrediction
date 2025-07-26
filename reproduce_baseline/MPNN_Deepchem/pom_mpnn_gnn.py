import torch.nn as nn
from dgl.nn.pytorch import NNConv
from dgllife.model.gnn import MPNNGNN


class CustomMPNNGNN(MPNNGNN):
   
    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum'):
        super(CustomMPNNGNN,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing)

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats), nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats))
        self.gnn_layer = NNConv(in_feats=node_out_feats,
                                out_feats=node_out_feats,
                                edge_func=edge_network,
                                aggregator_type=message_aggregator_type,
                                residual=residual)
