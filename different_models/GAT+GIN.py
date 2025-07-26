import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, global_add_pool

class ReadoutLayer(nn.Module):
    def __init__(self):
        super(ReadoutLayer, self).__init__()

    def forward(self, x, batch):
        return global_add_pool(x, batch)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.out = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.out(x)

def make_gin_mlp(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU()
    )

class OdorClassifier(nn.Module):
    def __init__(self, num_tasks, mlp_dims=[100, 70], dropout_p=0.1):
        super(OdorClassifier, self).__init__()
        input_dim = 15  # node feature size

        # GIN layers
        self.gin1 = GINConv(make_gin_mlp(input_dim, 20))
        self.bn1 = nn.BatchNorm1d(20)

        self.gin2 = GINConv(make_gin_mlp(20, 27))
        self.bn2 = nn.BatchNorm1d(27)

        # GAT layers (multi-head attention optional)
        self.gat1 = GATConv(27, 36, heads=1, dropout=dropout_p)
        self.bn3 = nn.BatchNorm1d(36)

        self.gat2 = GATConv(36, 92, heads=1, dropout=dropout_p)
        self.bn4 = nn.BatchNorm1d(92)

        self.readout1 = ReadoutLayer()
        self.readout2 = ReadoutLayer()
        self.readout3 = ReadoutLayer()
        self.readout4 = ReadoutLayer()

        self.mlp = MLPClassifier(input_dim=20 + 27 + 36 + 92 + 26, hidden_dims=mlp_dims, output_dim=num_tasks)

    def forward(self, data, return_projections=False):
        x, edge_index, mol_features, batch = data.x, data.edge_index, data.mol_features, data.batch

        # GIN layer 1
        x1 = F.selu(self.bn1(self.gin1(x, edge_index)))
        r1 = self.readout1(x1, batch)

        # GIN layer 2
        x2 = F.selu(self.bn2(self.gin2(x1, edge_index)))
        r2 = self.readout2(x2, batch)

        # GAT layer 1
        x3 = F.elu(self.bn3(self.gat1(x2, edge_index)))
        r3 = self.readout3(x3, batch)

        # GAT layer 2
        x4 = F.elu(self.bn4(self.gat2(x3, edge_index)))
        r4 = self.readout4(x4, batch)

        r_cat = torch.cat([r1, r2, r3, r4], dim=1)
        combined = torch.cat([r_cat, mol_features], dim=1)
        output = self.mlp(combined)

        self.saved_projections = {
            'readout1': r1.detach().cpu(),
            'readout2': r2.detach().cpu(),
            'readout3': r3.detach().cpu(),
            'readout4': r4.detach().cpu()
        }

        return (output, self.saved_projections) if return_projections else output
