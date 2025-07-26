import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
from torch_geometric.nn import GATConv, GraphConv, GCNConv, global_max_pool, global_add_pool, GINConv

# Aggregate node features into graph representation
class ReadoutLayer(nn.Module):
    def __init__(self):
        super(ReadoutLayer, self).__init__()

    def forward(self, x, batch):
        pooled = global_add_pool(x, batch)
        return pooled

# MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.out(x)

# Full model
class OdorClassifier(nn.Module):
    def __init__(self, num_tasks, mlp_dims=[100, 70], dropout_p=0.1):
        super(OdorClassifier, self).__init__()

        self.dropout_p = dropout_p

        self.conv1 = GCNConv(15, 20)
        self.bn1 = nn.BatchNorm1d(20)

        self.conv2 = GCNConv(20, 27)
        self.bn2 = nn.BatchNorm1d(27)

        self.conv3 = GCNConv(27, 36)
        self.bn3 = nn.BatchNorm1d(36)

        self.conv4 = GCNConv(36, 92)
        self.bn4 = nn.BatchNorm1d(92)

        # self.conv1 = GATConv(15, 40, heads=2, concat=True)
        # self.bn1 = nn.BatchNorm1d(80)

        # self.conv2 = GATConv(80, 57, heads=2, concat=True)
        # self.bn2 = nn.BatchNorm1d(114)

        # self.conv3 = GATConv(114, 65, heads=2, concat=True)
        # self.bn3 = nn.BatchNorm1d(130)

        # self.conv4 = GATConv(130, 92, heads=1, concat=False)
        # self.bn4 = nn.BatchNorm1d(92)

        # self.conv1 = GCNConv(15, 64)
        # # self.bn1 = nn.BatchNorm1d(20)
        # self.pool1 = SAGPooling(64, ratio=0.7, GNN= GCNConv)

        # self.conv2 = GCNConv(64, 128)
        # # self.bn2 = nn.BatchNorm1d(27)
        # self.pool2 = SAGPooling(128, ratio=0.7, GNN= GCNConv)

        # self.readout1 = Set2Set(20, processing_steps=3)
        # self.readout2 = Set2Set(129, processing_steps=3)
        self.readout1 = ReadoutLayer()
        self.readout2 = ReadoutLayer()
        self.readout3 = ReadoutLayer()
        self.readout4 = ReadoutLayer()

        self.mlp = MLPClassifier(input_dim= 20 + 27 + 36 + 92 + 55, hidden_dims=mlp_dims, output_dim=num_tasks) #258+26

    def forward(self, data, return_projections=False):
        x, edge_index, mol_features, batch = data.x, data.edge_index, data.mol_features, data.batch

        # x = F.relu(self.conv1(x, edge_index))
        # x, edge_index, _, batch, _, _= self.pool1(x, edge_index, None, batch)
        # r1 = self.readout1(x, batch)

        # x = F.relu(self.conv2(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # r2 = self.readout2(x, batch)

        x1 = self.conv1(x, edge_index)
        x1 = F.selu(self.bn1(x1))
        r1 = self.readout1(x1, batch)

        x2 = self.conv2(x1, edge_index)
        x2 = F.selu(self.bn2(x2))
        r2 = self.readout2(x2, batch)

        x3 = self.conv3(x2, edge_index)
        x3 = F.selu(self.bn3(x3))
        r3 = self.readout3(x3, batch)

        x4 = self.conv4(x3, edge_index)
        x4 = F.selu(self.bn4(x4))
        r4 = self.readout4(x4, batch)

        # Concate 
        r_cat = torch.cat([r1, r2, r3, r4], dim=1)
        
        # Final MLP input
        combined = torch.cat([r_cat, mol_features], dim=1)
        output = self.mlp(combined)

        # Save for optional inspection
        self.saved_projections = {
            'readout1': r1.detach().cpu(),
            'readout2': r2.detach().cpu(), 
            'readout3': r3.detach().cpu(),
            'readout4': r4.detach().cpu()
        }

        return (output, self.saved_projections) if return_projections else output
