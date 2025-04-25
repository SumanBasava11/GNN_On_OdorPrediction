import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from Featurizer.from_smiles import from_smiles

# Dataset class
class OdorDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles_list = smiles_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        data = from_smiles(smiles)
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
    # Separate the graphs and labels
    graphs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    batched_graphs = MoleculeDataBatch.from_data_list(graphs)
    
    # Stack the labels
    batched_labels = torch.stack(labels)
    
    return batched_graphs, batched_labels
