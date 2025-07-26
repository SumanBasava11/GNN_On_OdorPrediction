import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Callable, Dict
import torch.nn as nn
from dgl.nn.pytorch import NNConv
from dgllife.model.gnn import MPNNGNN

from deepchem.models.losses import Loss, L2Loss
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import Optimizer, LearningRateSchedule
from deepchem.models.optimizers import Adam

try:
    import dgl
    from dgl import DGLGraph
    from dgl.nn.pytorch import Set2Set
except (ImportError, ModuleNotFoundError):
    raise ImportError('This module requires dgl and dgllife')

import tempfile
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Iterator
from deepchem.data.datasets import DiskDataset, NumpyDataset
from skmultilearn.model_selection import IterativeStratification
from deepchem.splits import Splitter


def get_class_imbalance_ratio(dataset: DiskDataset) -> List:

    df: pd.DataFrame = pd.DataFrame(dataset.y)
    class_counts: np.ndarray = df.sum().to_numpy()
    max_count: int = max(class_counts)
    class_imbalance_ratio: List = (class_counts / max_count).tolist()
    return class_imbalance_ratio

class CustomMultiLabelLoss(Loss):

    def __init__(self,
                 class_imbalance_ratio: Optional[List] = None,
                 loss_aggr_type: str = 'sum',
                 device: Optional[str] = None):
        
        super(CustomMultiLabelLoss, self).__init__()
        if class_imbalance_ratio is None:
            print(Warning("No class imbalance ratio provided!"))
            self.class_imbalance_ratio: Optional[torch.Tensor] = None
        else:
            self.class_imbalance_ratio = torch.Tensor(class_imbalance_ratio)

        if loss_aggr_type not in ['sum', 'mean']:
            raise ValueError(f"Invalid loss aggregate type: {loss_aggr_type}")
        self.loss_aggr_type: str = loss_aggr_type

        if device is not None:
            if self.class_imbalance_ratio is not None:
                self.class_imbalance_ratio = self.class_imbalance_ratio.to(
                    device)

    def _create_pytorch_loss(
            self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns loss function for pytorch backend
        """
        ce_loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(
            reduction='none')

        def loss(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            
            # Convert (batch_size, tasks, classes)
            # to (batch_size, classes, tasks)
            # CrossEntropyLoss only supports (batch_size, classes, tasks)
            # This is for API consistency
            if len(output.shape) == 3:
                output = output.permute(0, 2, 1)

            if len(labels.shape) == len(output.shape):
                labels = labels.squeeze(-1)

            # handle multilabel
            # output shape => (batch_size, classes=1, tasks)
            # binary_output shape => (batch_size, classes=2, tasks)
            # where now we have (1 - probabilities) for ce loss calculation
            probabilities: torch.Tensor = output[:, 0, :]
            complement_probabilities: torch.Tensor = 1 - probabilities
            binary_output: torch.Tensor = torch.stack(
                [complement_probabilities, probabilities], dim=1)

            ce_loss: torch.Tensor = ce_loss_fn(binary_output, labels.long())

            if self.class_imbalance_ratio is None:
                if self.loss_aggr_type == 'sum':
                    loss: torch.Tensor = ce_loss.sum(dim=1)
                else:
                    loss = ce_loss.mean(dim=1)
            else:
                balancing_factors: torch.Tensor = torch.log(
                    1 + self.class_imbalance_ratio)

                # loss being weighted by a factor of
                # log(1+ class_imbalance_ratio)
                balanced_losses: torch.Tensor = torch.mul(
                    ce_loss, balancing_factors)

                if self.loss_aggr_type == 'sum':
                    # sum balanced loss across all tasks;
                    # shape => (batch_size)
                    loss = balanced_losses.sum(dim=1)
                else:
                    # mean balanced loss across all tasks;
                    # shape => (batch_size)
                    loss = balanced_losses.mean(dim=1)

            # duplicate loss across all tasks in a batch;
            # shape => (batch_size, n_tasks)
            # This is for API consistency
            return loss.unsqueeze(-1).repeat(1, output.shape[-1])

        return loss

class CustomPositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        d_input: int = 1024,
        d_hidden_list: List = [1024],
        d_output: int = 1024,
        activation: str = 'leakyrelu',
        dropout_p: float = 0.0,
        dropout_at_input_no_act: bool = False,
        batch_norm: bool = True,
    ):
        
        super(CustomPositionwiseFeedForward, self).__init__()

        self.dropout_at_input_no_act: bool = dropout_at_input_no_act
        self.batch_norm: bool = batch_norm

        self.activation: Callable[[Any], Any]
        if activation == 'relu':
            self.activation = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        elif activation == 'linear':
            self.activation = lambda x: x

        d_output = d_output if d_output != 0 else d_input

        # Set n_layers
        self.n_layers: int = len(d_hidden_list) + 1

        # Set linear layers
        if self.n_layers == 1:
            linears: List = [nn.Linear(d_input, d_output)]

        else:
            linears = [nn.Linear(d_input, d_hidden_list[0])]
            for idx in range(1, len(d_hidden_list)):
                linears.append(
                    nn.Linear(d_hidden_list[idx - 1], d_hidden_list[idx]))
            linears.append(nn.Linear(d_hidden_list[-1], d_output))

        self.linears: nn.ModuleList = nn.ModuleList(linears)
        dropout_layer: nn.Dropout = nn.Dropout(dropout_p)
        self.dropout_p: nn.ModuleList = nn.ModuleList(
            [dropout_layer for _ in range(self.n_layers)])

        if batch_norm:
            batchnorms: List = [
                nn.BatchNorm1d(d_hidden_list[idx])
                for idx in range(len(d_hidden_list))
            ]
            self.batchnorms: nn.ModuleList = nn.ModuleList(batchnorms)

    def forward(self, x: torch.Tensor) -> List[Optional[torch.Tensor]]:
    
        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return [None, self.linears[0](self.dropout_p[0](x))]
            else:
                return [
                    None,
                    self.dropout_p[0](self.activation(self.linears[0](x)))
                ]

        else:
            if self.dropout_at_input_no_act:
                x = self.dropout_p[-1](x)

            if self.batch_norm:
                for i in range(self.n_layers - 2):
                    x = self.dropout_p[i](self.activation(self.batchnorms[i](
                        self.linears[i](x))))

                embeddings: torch.Tensor = self.linears[self.n_layers - 2](x)
                x = self.dropout_p[self.n_layers - 2](self.activation(
                    self.batchnorms[self.n_layers - 2](embeddings)))
            else:
                for i in range(self.n_layers - 2):
                    x = self.dropout_p[i](self.activation(self.linears[i](x)))

                embeddings = self.linears[self.n_layers - 2](x)
                x = self.dropout_p[self.n_layers - 2](
                    self.activation(embeddings))

            output: torch.Tensor = self.linears[-1](x)
            return [embeddings, output]

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


class MPNNPOM(nn.Module):
    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 mode: str = 'classification',
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 nfeat_name: str = 'x',
                 efeat_name: str = 'edge_attr',
                 readout_type: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True):

        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        super(MPNNPOM, self).__init__()

        self.n_tasks: int = n_tasks
        self.mode: str = mode
        self.n_classes: int = n_classes
        self.nfeat_name: str = nfeat_name
        self.efeat_name: str = efeat_name
        self.readout_type: str = readout_type
        self.ffn_embeddings: int = ffn_embeddings
        self.ffn_activation: str = ffn_activation
        self.ffn_dropout_p: float = ffn_dropout_p

        if mode == 'classification':
            self.ffn_output: int = n_tasks * n_classes
        else:
            self.ffn_output = n_tasks

        self.mpnn: nn.Module = CustomMPNNGNN(
            node_in_feats=number_atom_features,
            node_out_feats=node_out_feats,
            edge_in_feats=number_bond_features,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type)

        self.project_edge_feats: nn.Module = nn.Sequential(
            nn.Linear(number_bond_features, edge_out_feats), nn.ReLU())

        if self.readout_type == 'set2set':
            self.readout_set2set: nn.Module = Set2Set(
                input_dim=node_out_feats + edge_out_feats,
                n_iters=num_step_set2set,
                n_layers=num_layer_set2set)
            ffn_input: int = 2 * (node_out_feats + edge_out_feats)
        elif self.readout_type == 'global_sum_pooling':
            ffn_input = node_out_feats + edge_out_feats
        else:
            raise Exception("readout_type invalid")

        if ffn_embeddings is not None:
            d_hidden_list: List = ffn_hidden_list + [ffn_embeddings]

        self.ffn: nn.Module = CustomPositionwiseFeedForward(
            d_input=ffn_input,
            d_hidden_list=d_hidden_list,
            d_output=self.ffn_output,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act)

    def _readout(self, g: DGLGraph, node_encodings: torch.Tensor,
                 edge_feats: torch.Tensor) -> torch.Tensor:

        g.ndata['node_emb'] = node_encodings
        g.edata['edge_emb'] = self.project_edge_feats(edge_feats)

        def message_func(edges) -> Dict:
            src_msg: torch.Tensor = torch.cat(
                (edges.src['node_emb'], edges.data['edge_emb']), dim=1)
            return {'src_msg': src_msg}

        def reduce_func(nodes) -> Dict:
            src_msg_sum: torch.Tensor = torch.sum(nodes.mailbox['src_msg'],
                                                  dim=1)
            return {'src_msg_sum': src_msg_sum}

        # radius 0 combination to fold atom and bond embeddings together
        g.send_and_recv(g.edges(),
                        message_func=message_func,
                        reduce_func=reduce_func)

        if self.readout_type == 'set2set':
            batch_mol_hidden_states: torch.Tensor = self.readout_set2set(
                g, g.ndata['src_msg_sum'])
        elif self.readout_type == 'global_sum_pooling':
            batch_mol_hidden_states = dgl.sum_nodes(g, 'src_msg_sum')

        # batch_size x (node_out_feats + edge_out_feats)
        return batch_mol_hidden_states

    def forward(
        self, g: DGLGraph
    ) -> Union[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        
        node_feats: torch.Tensor = g.ndata[self.nfeat_name]
        edge_feats: torch.Tensor = g.edata[self.efeat_name]

        node_encodings: torch.Tensor = self.mpnn(g, node_feats, edge_feats)

        molecular_encodings: torch.Tensor = self._readout(
            g, node_encodings, edge_feats)
        if self.readout_type == 'global_sum_pooling':
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        embeddings: torch.Tensor
        out: torch.Tensor
        embeddings, out = self.ffn(molecular_encodings)

        if self.mode == 'classification':
            if self.n_tasks == 1:
                logits: torch.Tensor = out.view(-1, self.n_classes)
            else:
                logits = out.view(-1, self.n_tasks, self.n_classes)
            proba: torch.Tensor = F.sigmoid(
                logits)  # (batch, n_tasks, classes)
            if self.n_classes == 1:
                proba = proba.squeeze(-1)  # (batch, n_tasks)
            return proba, logits, embeddings
        else:
            return out


class MPNNPOMModel(TorchModel):

    def __init__(self,
                 n_tasks: int,
                 class_imbalance_ratio: Optional[List] = None,
                 loss_aggr_type: str = 'sum',
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 batch_size: int = 100,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 edge_out_feats: int = 64,
                 num_step_message_passing: int = 3,
                 mpnn_residual: bool = True,
                 message_aggregator_type: str = 'sum',
                 mode: str = 'regression',
                 number_atom_features: int = 134,
                 number_bond_features: int = 6,
                 n_classes: int = 1,
                 readout_type: str = 'set2set',
                 num_step_set2set: int = 6,
                 num_layer_set2set: int = 3,
                 ffn_hidden_list: List = [300],
                 ffn_embeddings: int = 256,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 weight_decay: float = 1e-5,
                 self_loop: bool = False,
                 optimizer_name: str = 'adam',
                 device_name: Optional[str] = None,
                 **kwargs):
        
        model: nn.Module = MPNNPOM(
            n_tasks=n_tasks,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            edge_out_feats=edge_out_feats,
            num_step_message_passing=num_step_message_passing,
            mpnn_residual=mpnn_residual,
            message_aggregator_type=message_aggregator_type,
            mode=mode,
            number_atom_features=number_atom_features,
            number_bond_features=number_bond_features,
            n_classes=n_classes,
            readout_type=readout_type,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set,
            ffn_hidden_list=ffn_hidden_list,
            ffn_embeddings=ffn_embeddings,
            ffn_activation=ffn_activation,
            ffn_dropout_p=ffn_dropout_p,
            ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act)

        if class_imbalance_ratio and (len(class_imbalance_ratio) != n_tasks):
            raise Exception("size of class_imbalance_ratio \
                            should be equal to n_tasks")
        
        if mode == 'regression':
            loss: Loss = L2Loss()
            output_types: List = ['prediction']
        else:
            loss = CustomMultiLabelLoss(
                class_imbalance_ratio=class_imbalance_ratio,
                loss_aggr_type=loss_aggr_type,
                device=device_name)
            output_types = ['prediction', 'loss', 'embedding']

        optimizer = Adam(learning_rate=0.001)
        # optimizer.learning_rate = learning_rate
        if device_name is not None:
            device: Optional[torch.device] = torch.device(device_name)
        else:
            device = None
        super(MPNNPOMModel, self).__init__(model,
                                           loss=loss,
                                           output_types=output_types,
                                           optimizer=optimizer,
                                           learning_rate=learning_rate,
                                           batch_size=batch_size,
                                           device=device,
                                           **kwargs)

        self.weight_decay: float = weight_decay
        self._self_loop: bool = self_loop
        self.regularization_loss: Callable = self._regularization_loss

    def _regularization_loss(self) -> torch.Tensor:
    
        l1_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        l2_regularization: torch.Tensor = torch.tensor(0., requires_grad=True)
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                l1_regularization = l1_regularization + torch.norm(param, p=1)
                l2_regularization = l2_regularization + torch.norm(param, p=2)
        l1_norm: torch.Tensor = self.weight_decay * l1_regularization
        l2_norm: torch.Tensor = self.weight_decay * l2_regularization
        return l1_norm + l2_norm

    def _prepare_batch(
        self, batch: Tuple[List, List, List]
    ) -> Tuple[DGLGraph, List[torch.Tensor], List[torch.Tensor]]:
    
        inputs: List
        labels: List
        weights: List

        inputs, labels, weights = batch
        dgl_graphs: List[DGLGraph] = [
            graph.to_dgl_graph(self_loop=self._self_loop)
            for graph in inputs[0]
        ]
        g: DGLGraph = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(MPNNPOMModel, self)._prepare_batch(
            ([], labels, weights))
        return g, labels, weights