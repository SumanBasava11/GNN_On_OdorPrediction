import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, global_max_pool, GATConv, GINEConv, GraphConv

class ReadoutLayer(nn.Module):
    def __init__(self):
        super(ReadoutLayer, self).__init__()

    def forward(self, x, batch):
        return global_max_pool(x, batch)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims,  output_dim, dropout=0.3):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.out = nn.Linear(hidden_dims[1], output_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)
    
# Helper function to create the MLP for GINConv
def make_gin_mlp(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU()
    )

def edge_mlp(edge_dim, hidden_dim):
    # MLP for edge features transformation
    return nn.Sequential(
        nn.Linear(edge_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU()
    )

class OdorGCNUNet(nn.Module):
    def __init__(self, num_tasks, mlp_dims=[100, 70],  alpha=0.5, dropout=0.3):
        super(OdorGCNUNet, self).__init__()
       
        self.input_dim = 134
        self.edge_dim = 6
        self.alpha = alpha
        self.edge_hidden_dim = 150 
        self.edge_mlp = edge_mlp(self.edge_dim, self.edge_hidden_dim)
        self.dropout = dropout

        # ----- Encoder -----
        self.conv1 = GINEConv(nn = make_gin_mlp(self.input_dim, 150), edge_dim=self.edge_hidden_dim)
        # self.pool_ratio1 = nn.Parameter(torch.tensor(0.7))
        self.pool1 = SAGPooling(150, ratio=0.8, GNN = GraphConv)

        self.conv2 = GINEConv(nn= make_gin_mlp(150, 164), edge_dim=self.edge_hidden_dim)
        # self.pool_ratio2 = nn.Parameter(torch.tensor(0.7))
        self.pool2 = SAGPooling(164, ratio=0.8, GNN = GraphConv)

        self.conv3 = GINEConv(nn= make_gin_mlp(164, 178), edge_dim=self.edge_hidden_dim)
        # self.pool_ratio3 = nn.Parameter(torch.tensor(0.7))
        self.pool3 = SAGPooling(178, ratio=0.8, GNN = GraphConv)

        self.conv4 = GINEConv(nn= make_gin_mlp(178, 200), edge_dim=self.edge_hidden_dim)

        self.readout2 = ReadoutLayer()
        self.readout3 = ReadoutLayer()
        self.readout4 = ReadoutLayer()

        # Graph Classification Head
        self.graph_mlp = MLPClassifier(input_dim=(164 + 178 + 200),
                                       hidden_dims=mlp_dims, output_dim=num_tasks)

        # ----- Decoder with Skip Connections -----
        self.unconv3 = GINEConv(make_gin_mlp(200, 178), edge_dim=self.edge_hidden_dim)
        self.unconv2 = GINEConv(make_gin_mlp(178, 164), edge_dim=self.edge_hidden_dim)
        self.unconv1 = GINEConv(make_gin_mlp(164, 150), edge_dim=self.edge_hidden_dim)
        self.node_predictor = nn.Linear(150, self.input_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # ----- Encoder Path -----
        edge_attr_transformed = self.edge_mlp(edge_attr)

        # conv1 on original graph
        x1 = F.relu(self.conv1(x, edge_index, edge_attr_transformed))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1_p, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(x1, edge_index, edge_attr, batch)    # Pool 1

        #  Transform pooled edge attributes
        edge_attr1_transformed = self.edge_mlp(edge_attr1)

        x2 = F.relu(self.conv2(x1_p, edge_index1, edge_attr1_transformed))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2_p, edge_index2, edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index1, edge_attr1, batch1)
        r2 = self.readout2(x2_p, batch2)

        edge_attr2_transformed = self.edge_mlp(edge_attr2)
        x3 = F.relu(self.conv3(x2_p, edge_index2, edge_attr2_transformed))
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        x3_p, edge_index3,  edge_attr3, batch3, perm3, score3 = self.pool3(x3, edge_index2, edge_attr2, batch2)
        r3 = self.readout3(x3_p, batch3)

        edge_attr3_transformed = self.edge_mlp(edge_attr3)
        x4 = F.relu(self.conv4(x3_p, edge_index3, edge_attr3_transformed))
        x4 = F.dropout(x4, p=self.dropout, training=self.training)
        r4 = self.readout4(x4, batch3)

        # ----- Graph-Level Output -----
        graph_embed = torch.cat([r2, r3, r4], dim=1)
        graph_out = self.graph_mlp(graph_embed)

        # ----- Decoder Path with Skip Connections -----
        # Decoder with unpooling (inverse permutation)
        x_un3 = F.relu(self.unconv3(x4, edge_index3, edge_attr3_transformed))
        x_un3 = F.dropout(x_un3, p=self.dropout, training=self.training)
        x_un3_full = torch.zeros_like(x3)
        x_un3_full[perm3] = x_un3
        x_un3_full += x3

        x_un2 = F.relu(self.unconv2(x_un3_full, edge_index2, edge_attr2_transformed))
        x_un2 = F.dropout(x_un2, p=self.dropout, training=self.training)
        x_un2_full = torch.zeros_like(x2)
        x_un2_full[perm2] = x_un2
        x_un2_full += x2

        x_un1 = F.relu(self.unconv1(x_un2_full, edge_index1, edge_attr1_transformed))
        x_un1 = F.dropout(x_un1, p=self.dropout, training=self.training)
        x_un1_full = torch.zeros_like(x1)
        x_un1_full[perm1] = x_un1
        x_un1_full += x1

        node_output = self.node_predictor(x_un1_full)

        return graph_out, node_output
    
