import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

# Readout Layer
class ReadoutLayer(nn.Module):
    def __init__(self, in_channels=55, out_channels=175):
        super(ReadoutLayer, self).__init__()
        self.global_pool = global_add_pool

    def forward(self, x, batch):
        # Global sum pooling
        # Apply softmax to node features across feature dimension
        # x = F.softmax(x, dim=1)
        x = self.global_pool(x, batch)
        return x

# Fully-Connected Neural Network (MLP) Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

        self.dropout = nn.Dropout(0.40)
        self.out = nn.Linear(hidden_dims[1], 163)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.out(x) 
        # x = torch.sigmoid(self.out(x)) 
        return x


# Full Model (GCN + Readout + MLP)
class OdorClassifier(nn.Module):
    def __init__(self, num_tasks, readout_dim=175, mlp_dims=[96, 63]):
        super(OdorClassifier, self).__init__()

        # Define 4 GCN layers
        self.gcn1 = GCNConv(23, 55)
        self.gcn2 = GCNConv(55, 67)
        self.gcn3 = GCNConv(67, 75)
        self.gcn4 = GCNConv(75, 85)

        # 4 readouts, one for each layer
        self.readout1 = ReadoutLayer()
        self.readout2 = ReadoutLayer()
        self.readout3 = ReadoutLayer()
        self.readout4 = ReadoutLayer()

        # MLP Classifier
        self.mlp = MLPClassifier(282 + 10, mlp_dims, 163)

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

        x = torch.cat([r1, r2, r3, r4], dim=1)
        
        # Get the batch size from x
        batch_size = x.size(0)
        
        # Concatenate molecular features
        x = torch.cat([x, mol_features], dim=1)

        # MLP Classifier
        return self.mlp(x)