import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

# Readout Layer
class ReadoutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReadoutLayer, self).__init__()
        self.global_pool = global_add_pool
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, batch):
        # Global sum pooling
        x = self.global_pool(x, batch)
        return self.fc(x)

# Fully-Connected Neural Network (MLP) Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

        self.dropout = nn.Dropout(0.47)
        self.out = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.out(x))


# Full Model (GCN + Readout + MLP)
class OdorClassifier(nn.Module):
    def __init__(self, num_tasks, readout_dim=175, mlp_dims=[96, 63]):
        super(OdorClassifier, self).__init__()

        # Define 4 GCN layers
        self.gcn1 = GCNConv(48, 55)
        self.gcn2 = GCNConv(55, 67)
        self.gcn3 = GCNConv(67, 70)
        self.gcn4 = GCNConv(70, 70)

        # 4 readouts, one for each layer
        self.readout1 = ReadoutLayer(55, readout_dim)
        self.readout2 = ReadoutLayer(67, readout_dim)
        self.readout3 = ReadoutLayer(70, readout_dim)
        self.readout4 = ReadoutLayer(70, readout_dim)

        # MLP Classifier
        self.mlp = MLPClassifier(readout_dim + 10, mlp_dims, num_tasks)

    def forward(self, data):
        x, edge_index, mol_features, batch = data.x, data.edge_index, data.mol_features, data.batch

        x1 = self.gcn1(x, edge_index)
        r1 = self.readout1(x1, batch)

        x2 = self.gcn2(x1, edge_index)
        r2 = self.readout2(x2, batch)

        x3 = self.gcn3(x2, edge_index)
        r3 = self.readout3(x3, batch)

        x4 = self.gcn4(x3, edge_index)
        r4 = self.readout4(x4, batch)

        # Sum all 4 readouts
        x = r1 + r2 + r3 + r4  # Shape: (batch_size, 175)

        # Get the batch size from x
        batch_size = x.size(0)
        
        # Concatenate molecular features
        x = torch.cat([x, mol_features], dim=1)

        # MLP Classifier
        return self.mlp(x)