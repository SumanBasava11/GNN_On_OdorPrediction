import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from Featurizer.from_smiles import from_smiles
# from reproduce_baseline.MPNN_Deepchem.GraphFeaturizer_deepchem import GraphFeaturizer

# Dataset class
class OdorDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles_list = smiles_list
        self.labels = labels
        # self.featurizer = GraphFeaturizer() 

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        data = from_smiles(smiles)
        # data = self.featurizer.featurize(smiles)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return data, label

# Molecule Feature Batching
class MoleculeDataBatch(Batch):
    @staticmethod
    def from_data_list(data_list):
        batch = Batch.from_data_list(data_list)
        
        # Handle molecular features separately
        mol_feats = torch.stack([d.mol_features for d in data_list])
        batch.mol_features = mol_feats
        
        return batch

# Custom collate function for PyTorch Geometric data
def collate_fn(batch):
    graphs = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    # Debug: check for None
    for i, g in enumerate(graphs):
        if g is None:
            print(f"Warning: graph at index {i} is None!")
    graphs = [g for g in graphs if g is not None]
    
    if len(graphs) == 0:
        raise ValueError("No valid graphs in batch!")
    
    batched_graphs = MoleculeDataBatch.from_data_list(graphs)
    # batched_graphs = Batch.from_data_list(graphs) 
    return batched_graphs, labels