import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool

class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__(aggr='max')
        self.conv = nn.Linear(in_channels, out_channels)
        self.activation = nn.SELU()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.conv(x_j)

    def update(self, aggr_out):
        return self.activation(aggr_out)


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
        self.gcn1 = GCNLayer(48, 20)
        self.gcn2 = GCNLayer(20, 27)
        self.gcn3 = GCNLayer(27, 36)
        self.gcn4 = GCNLayer(36, 36)

        # 4 readouts, one for each layer
        self.readout1 = ReadoutLayer(20, readout_dim)
        self.readout2 = ReadoutLayer(27, readout_dim)
        self.readout3 = ReadoutLayer(36, readout_dim)
        self.readout4 = ReadoutLayer(36, readout_dim)

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

        # Extract and batch molecular features
        if hasattr(data, 'mol_features'):
            if batch_size > 1:
                # Get unique molecules in the batch
                unique_batch_indices = torch.unique(batch) if batch is not None else torch.tensor([0])
                num_graphs = len(unique_batch_indices)      
                # Reshape mol_features to match batch size by repeating if necessary
                if data.mol_features.dim() == 1:
                    # If it's a single feature vector, expand to batch size
                    mol_features = data.mol_features.unsqueeze(0).expand(batch_size, -1)
                elif data.mol_features.size(0) == 1:
                    # If it's a single sample with batch dim, expand to batch size
                    mol_features = data.mol_features.expand(batch_size, -1)
                elif data.mol_features.size(0) == batch_size:
                    # If already correct size, use as is
                    mol_features = data.mol_features
                else:
                    # Try to reshape based on number of graphs
                    if num_graphs == batch_size:
                        mol_features = data.mol_features
                    else:
                        print(f"Warning: Molecular features shape mismatch. Expected batch size {batch_size}, got {data.mol_features.size(0)}")
                        mol_features = data.mol_features.unsqueeze(0).expand(batch_size, -1)
            else:
                # Single graph case
                mol_features = data.mol_features.unsqueeze(0) if data.mol_features.dim() == 1 else data.mol_features
        else:
            # If no molecular features, use zeros
            mol_features = torch.zeros(batch_size, 10, device=x.device)

        # Concatenate molecular features
        x = torch.cat([x, mol_features], dim=1)

        # MLP Classifier
        return self.mlp(x)