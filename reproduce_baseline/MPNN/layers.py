import math
from math import pi as PI
import numpy as np
from typing import Any, Tuple, Optional, Sequence, List, Union, Callable, Dict, TypedDict
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError('These classes require PyTorch to be installed.')
from deepchem.utils.pytorch_utils import unsorted_segment_sum
from torch.nn import init as initializers

def unsorted_segment_max(data: torch.Tensor, segment_ids: torch.Tensor,
                         num_segments: int) -> torch.Tensor:
    
    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have to be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError(
            "segment_ids should be the same size as dimension 0 of input.")

    # Initialize the tensor to hold the maximum values for each segment
    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.full(shape, float('-inf'), dtype=data.dtype)

    # Create an expanded segment_ids tensor to match data shape
    expanded_segment_ids = segment_ids.unsqueeze(-1).expand(-1, *data.shape[1:])

    # Update the maximum values for each segment
    for i in range(num_segments):
        mask = expanded_segment_ids == i
        tensor[i] = torch.max(data.masked_fill(~mask, float('-inf')), dim=0)[0]

    return tensor

class GraphConv(nn.Module):
    
    def __init__(self,
                 out_channel: int,
                 number_input_features: int,
                 min_deg: int = 0,
                 max_deg: int = 10,
                 activation_fn: Optional[Callable] = None,
                 **kwargs):
       
        super(GraphConv, self).__init__(**kwargs)
        self.out_channel: int = out_channel
        self.min_degree: int = min_deg
        self.max_degree: int = max_deg
        self.number_input_features: int = number_input_features
        self.activation_fn: Optional[Callable] = activation_fn

        # Generate the nb_affine weights and biases
        num_deg: int = 2 * self.max_degree + (1 - self.min_degree)
        self.W_list: nn.ParameterList = nn.ParameterList([
            nn.Parameter(
                getattr(initializers,
                        'xavier_uniform_')(torch.empty(number_input_features,
                                                       self.out_channel)))
            for k in range(num_deg)
        ])
        self.b_list: nn.ParameterList = nn.ParameterList([
            nn.Parameter(
                getattr(initializers, 'zeros_')(torch.empty(self.out_channel,)))
            for k in range(num_deg)
        ])
        self.built = True

    def __repr__(self) -> str:
       
        # flake8: noqa
        return (
            f'{self.__class__.__name__}(out_channel:{self.out_channel},min_deg:{self.min_deg},max_deg:{self.max_deg},activation_fn:{self.activation_fn})'
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:

        # Extract atom_features
        atom_features: torch.Tensor = inputs[0]

        # Extract graph topology
        deg_slice: torch.Tensor = inputs[1]
        deg_adj_lists: List[torch.Tensor] = inputs[3:]

        W = iter(self.W_list)
        b = iter(self.b_list)

        # Sum all neighbors using adjacency matrix
        deg_summed: List[np.ndarray] = self.sum_neigh(atom_features,
                                                      deg_adj_lists)

        # Get collection of modified atom features
        new_rel_atoms_collection = []

        split_features: Tuple[torch.Tensor,
                              ...] = torch.split(atom_features,
                                                 (deg_slice[:, 1]).tolist())
        for deg in range(1, self.max_degree + 1):
            # Obtain relevant atoms for this degree
            rel_atoms: torch.Tensor = torch.from_numpy(deg_summed[deg - 1])

            # Get self atoms
            self_atoms: torch.Tensor = split_features[deg - self.min_degree]

            # Apply hidden affine to relevant atoms and append
            rel_out: torch.Tensor = torch.matmul(rel_atoms.type(torch.float32),
                                                 next(W)) + next(b)
            self_out: torch.Tensor = torch.matmul(
                self_atoms.type(torch.float32), next(W)) + next(b)
            out: torch.Tensor = rel_out + self_out
            new_rel_atoms_collection.append(
                torch.from_numpy(out.detach().numpy()))

        # Determine the min_deg=0 case
        if self.min_degree == 0:
            self_atoms = split_features[0]

            # Only use the self layer
            out = torch.matmul(self_atoms.type(torch.float32),
                               next(W)) + next(b)
            new_rel_atoms_collection.insert(
                0, torch.from_numpy(out.detach().numpy()))

        # Combine all atoms back into the list
        atom_features = torch.concat(new_rel_atoms_collection, 0)

        if self.activation_fn is not None:
            atom_features = self.activation_fn(atom_features)

        return atom_features

    def sum_neigh(self, atoms: torch.Tensor, deg_adj_lists) -> List[np.ndarray]:
        """Store the summed atoms by degree"""
        deg_summed = []

        for deg in range(1, self.max_degree + 1):
            gathered_atoms: torch.Tensor = atoms[deg_adj_lists[deg - 1]]
            # Sum along neighbors as well as self, and store
            summed_atoms: torch.Tensor = torch.sum(gathered_atoms, 1)
            deg_summed.append(summed_atoms.detach().numpy())

        return deg_summed


class GraphPool(nn.Module):

    def __init__(self, min_degree: int = 0, max_degree: int = 10, **kwargs):
        
        super(GraphPool, self).__init__(**kwargs)
        self.min_degree: int = min_degree
        self.max_degree: int = max_degree

    def get_config(self) -> str:
    
        # flake8: noqa
        return (
            f'{self.__class__.__name__}(min_degree:{self.min_degree},max_degree:{self.max_degree})'
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
       
        atom_features: torch.Tensor = inputs[0]
        deg_slice: torch.Tensor = inputs[1]
        deg_adj_lists: List[torch.Tensor] = inputs[3:]

        # Perform the mol gather
        deg_maxed = []

        split_features: Tuple[torch.Tensor,
                              ...] = torch.split(atom_features,
                                                 (deg_slice[:, 1]).tolist())
        for deg in range(1, self.max_degree + 1):
            # Get self atoms
            self_atoms: torch.Tensor = split_features[deg - self.min_degree]

            if deg_adj_lists[deg - 1].shape[0] == 0:
                # There are no neighbors of this degree, so just create an empty tensor directly.
                maxed_atoms: torch.Tensor = torch.zeros(
                    (0, self_atoms.shape[-1]))
                deg_maxed.append(maxed_atoms)
            else:
                # Expand dims
                self_atoms = torch.unsqueeze(self_atoms, 1)

                # always deg-1 for deg_adj_lists
                gathered_atoms: torch.Tensor = atom_features[deg_adj_lists[deg -
                                                                           1]]
                gathered_atoms = torch.concat([self_atoms, gathered_atoms], 1)

                max_atoms: tuple = torch.max(gathered_atoms, 1)
                deg_maxed.append(max_atoms[0])

        if self.min_degree == 0:
            self_atoms = split_features[0]
            deg_maxed.insert(0, self_atoms)

        return torch.concat(deg_maxed, 0)


class GraphGather(nn.Module):

    def __init__(self,
                 batch_size: int,
                 activation_fn: Optional[Callable] = None,
                 **kwargs):

        super(GraphGather, self).__init__(**kwargs)
        self.batch_size: int = batch_size
        self.activation_fn: Optional[Callable] = activation_fn

    def get_config(self) -> str:
    
        # flake8: noqa
        return (
            f'{self.__class__.__name__}(batch_size:{self.batch_size},activation_fn:{self.activation_fn})'
        )

    def forward(self, inputs: List[torch.Tensor]):

        atom_features: torch.Tensor = inputs[0]

        # Extract graph topology
        membership: torch.Tensor = inputs[2].to(torch.int64)

        assert self.batch_size > 1, "graph_gather requires batches larger than 1"

        sparse_reps: torch.Tensor = unsorted_segment_sum(
            atom_features, membership, self.batch_size)
        max_reps: torch.Tensor = unsorted_segment_max(atom_features, membership,
                                                      self.batch_size)
        mol_features: torch.Tensor = torch.concat([sparse_reps, max_reps], 1)

        if self.activation_fn is not None:
            mol_features = self.activation_fn(mol_features)
        return mol_features
