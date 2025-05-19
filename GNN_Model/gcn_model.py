import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.transforms import ToDense
data = ToDense(num_nodes=MAX_NODES)(data)

# Readout Layer: Includes projection to fixed output dimension (175)
class ReadoutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReadoutLayer, self).__init__()
        self.global_pool = global_add_pool
        self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, x, batch):
        x = self.global_pool(x, batch)              # [batch_size, in_channels]
        x = self.projection(x)                      # [batch_size, out_channels]
        return x

# 2-layer MLP Classifier with dropout at each layer
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

        self.dropout = nn.Dropout(0.30)
        self.out = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.out(x)

# Full Model: GCN + Readouts (projected to 175 dim) + Summation + MLP
class OdorClassifier(nn.Module):
    def __init__(self, num_tasks, mlp_dims=[96, 63]):
        super(OdorClassifier, self).__init__()
        self.num_tasks = num_tasks

        # GCN layers
        self.gcn1 = GCNConv(15, 20)
        self.bn_gcn1 = nn.BatchNorm1d(20)

        self.gcn2 = GCNConv(20, 27)
        self.bn_gcn2 = nn.BatchNorm1d(27)

        # Readout layers with projection to 175 dimensions
        self.readout1 = ReadoutLayer(in_channels=20, out_channels=175)
        self.readout2 = ReadoutLayer(in_channels=27, out_channels=175)

        # Final MLP: Input = 175 (readout sum) + 10 (molecular features)
        self.mlp = MLPClassifier(input_dim=175 + 10, hidden_dims=mlp_dims, output_dim=num_tasks)

        # Storage for external projection access
        self.saved_projections = {}

    def forward(self, data, return_projections=False):
        x, edge_index, mol_features, batch = data.x, data.edge_index, data.mol_features, data.batch

        # GCN layer 1 and readout
        x1 = self.gcn1(x, edge_index)
        x1 = F.relu(self.bn_gcn1(x1))
        r1 = self.readout1(x1, batch)

        # GCN layer 2 and readout
        x2 = self.gcn2(x1, edge_index)
        x2 = F.relu(self.bn_gcn2(x2))
        r2 = self.readout2(x2, batch)

        # Sum readouts from both layers
        r_sum = r1 + r2  # [batch_size, 175]

        # Optionally store projections
        self.saved_projections['readout1'] = r1.detach().cpu()
        self.saved_projections['readout2'] = r2.detach().cpu()

        # Combine with molecular features
        combined = torch.cat([r_sum, mol_features], dim=1)  # [batch_size, 185]

        # Final classification
        output = self.mlp(combined)

        if return_projections:
            return output, self.saved_projections
        return output
