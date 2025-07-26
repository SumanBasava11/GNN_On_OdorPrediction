import torch
import torch.nn as nn
import torch.nn.functional as F
# from deepchem.models.torch_models.layers import GraphConv, GraphGather, GraphPool
from reproduce_baseline.MPNN.layers import GraphConv, GraphGather, GraphPool

batch_size = 100
n_tasks = 138

class GraphConvModelTorch(nn.Module):
    def __init__(self, input_dim=111, hidden_dim=128, batch_norm=True):
        super(GraphConvModelTorch, self).__init__()

        self.gc1 = GraphConv(out_channel=hidden_dim, number_input_features=input_dim, activation_fn=nn.Tanh())
        self.bn1 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.pool1 = GraphPool()

        self.gc2 = GraphConv(out_channel=hidden_dim, number_input_features=hidden_dim, activation_fn=nn.Tanh())
        self.bn2 = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()
        self.pool2 = GraphPool()

        # self.dense1 = nn.Linear(128, 256)
        # self.act3 = nn.Tanh()
        # self.batch_norm3 = nn.BatchNorm1d(256)
        # self.readout = GraphGather(batch_size=batch_size, activation_fn=nn.Tanh())
        
        # self.dense2 = nn.Linear(512, n_tasks * 2)  
       
        # self.logits = lambda data: data.view(-1, n_tasks, 2)
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, inputs):
        x1 = self.gc1(inputs)
        bn1_output = self.bn1(x1)
        x1 = self.gp1([bn1_output] + inputs[1:])

        x2 = self.gc2([x1] + inputs[1:])
        bn2_output = self.bn2(x2)
        x2 = self.gp2([bn2_output] + inputs[1:])
        return x2

        # dense1_output = self.act3(self.dense1(x2))
        # bn3_output = self.bn3(dense1_output)
        # readout_output = self.readout([bn3_output] + inputs[1:])
        
        # dense2_output = self.dense2(readout_output)
        # logits_output = self.logits(dense2_output)
        # softmax_output = self.softmax(logits_output)
        # return softmax_output

class GraphDecoder(nn.Module):
    def __init__(self, batch_size, hidden_dim=128, readout_dim=512, n_tasks=138):
        super().__init__()
        self.readout = GraphGather(
            batch_size=batch_size,
            activation_fn=nn.Tanh()
        )
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(readout_dim, n_tasks * 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encoded, inputs):
        x = self.act1(self.bn1(self.fc1(encoded)))
        x = self.readout([x] + inputs[1:])
        x = self.fc2(x)
        x = x.view(-1, x.size(1) // 2, 2)  # reshape to (batch_size, n_tasks, 2)
        return self.softmax(x)
    
class DeepChemGraphClassifier(nn.Module):
    def __init__(self, batch_size=100, input_dim=111, n_tasks=138):
        super().__init__()
        self.encoder = GraphConvModelTorch(input_dim=input_dim)
        self.decoder = GraphDecoder(batch_size=batch_size, n_tasks=n_tasks)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        out = self.decoder(encoded, inputs)
        return out