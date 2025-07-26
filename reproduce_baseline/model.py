import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, SAGPooling, GraphConv, GCNConv, GATConv

# Aggregate node features into graph representation
class ReadoutLayer(nn.Module):
    def __init__(self):
        super(ReadoutLayer, self).__init__()

    def forward(self, x, batch):
        pooled = global_add_pool(x, batch)
        return pooled

# MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.drop1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.drop2 = nn.Dropout(p=dropout_p)

        self.out = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        return self.out(x)

# Helper function to create the MLP for GINConv
def make_gin_mlp(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
    )

# Full model with GINConv layers
class OdorClassifier(nn.Module):
    def __init__(self, num_tasks, mlp_dims=[100, 80], dropout_p=0.2):
        super(OdorClassifier, self).__init__()
        self.num_tasks = num_tasks
        input_dim = 15
        self.dropout_p = dropout_p

        self.conv1 = GINConv(make_gin_mlp(input_dim, 20))
        self.pool1 = SAGPooling(20, ratio=0.8, GNN = GraphConv)
        self.bn1 = nn.BatchNorm1d(20)
        self.drop1 = nn.Dropout(p=dropout_p)

        self.conv2 = GINConv(make_gin_mlp(20, 155))
        self.pool2 = SAGPooling(155, ratio=0.8, GNN = GraphConv)
        self.bn2 = nn.BatchNorm1d(155)
        self.drop2 = nn.Dropout(p=dropout_p)

        # self.conv3 = GINConv(make_gin_mlp(27, 36))
        # self.pool3 = SAGPooling(36, ratio=0.8, GNN = GraphConv)
        # self.bn3 = nn.BatchNorm1d(36)
        # self.drop3 = nn.Dropout(p=dropout_p)

        # self.conv4 = GINConv(make_gin_mlp(36, 92))
        # self.pool4 = SAGPooling(92, ratio=0.8, GNN = GraphConv)
        # self.bn4 = nn.BatchNorm1d(92)
        # self.drop4 = nn.Dropout(p=dropout_p)

        self.readout1 = ReadoutLayer()
        self.readout2 = ReadoutLayer()
        # self.readout3 = ReadoutLayer()
        # self.readout4 = ReadoutLayer()

        self.mlp = MLPClassifier(input_dim= 175+51, hidden_dims=mlp_dims, output_dim=num_tasks, dropout_p=dropout_p)
        self.thresholds = nn.Parameter(torch.zeros(num_tasks), requires_grad=True)
    
    def forward(self, data, return_projections=False):
        x, edge_index, mol_features, batch = data.x, data.edge_index, data.mol_features, data.batch

        x1 = self.conv1(x, edge_index)
        x1 = F.selu(self.bn1(x1))
        x1 = self.drop1(x1)
        x1, edge_index1, _, batch1, _, _ = self.pool1(x1, edge_index, None, batch)
        r1 = self.readout1(x1, batch1)
        # r1 = F.softmax(r1, dim=1)

        x2 = self.conv2(x1, edge_index1)
        x2 = F.selu(self.bn2(x2))
        x2 = self.drop2(x2)
        x2, edge_index2, _, batch2, _, _ = self.pool2(x2, edge_index1, None, batch1)
        r2 = self.readout2(x2, batch2)
        # r2 = F.softmax(r2, dim=1)

        # x3 = self.conv3(x2, edge_index2)
        # x3 = F.selu(self.bn3(x3))
        # x3 = self.drop3(x3)
        # x3, edge_index3, _, batch3, _, _ = self.pool3(x3, edge_index2, None, batch2)
        # r3 = self.readout3(x3, batch3)

        # x4 = self.conv4(x3, edge_index3)
        # x4 = F.selu(self.bn4(x4))
        # x4 = self.drop4(x4)
        # x4, edge_index4, _, batch4, _, _ = self.pool4(x4, edge_index3, None, batch3)
        # r4 = self.readout4(x4, batch4)

        # Concatenate pooled outputs from all layers
        r_cat = torch.cat([r1, r2], dim=1)

        # Concatenate with molecular features
        combined = torch.cat([r_cat, mol_features], dim=1)
        output = self.mlp(combined)

        # Save for optional inspection
        self.saved_projections = {
            'readout1': r1.detach().cpu(),
            'readout2': r2.detach().cpu()
            # 'readout3': r3.detach().cpu(),
            # 'readout4': r4.detach().cpu()
        }
        return (output, self.saved_projections) if return_projections else output
