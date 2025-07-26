import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from reproduce_baseline.MPNN_Deepchem.GraphFeaturizer_deepchem import GraphFeaturizer, GraphConvConstants
from deepchem.feat.graph_data import GraphData

def graphdata_to_pyg(data: GraphData) -> Data:
    node_feats = torch.tensor(data.node_features, dtype=torch.float)
    edge_index = torch.tensor(data.edge_index, dtype=torch.long)
    edge_feats = torch.tensor(data.edge_features, dtype=torch.float)

    pyg_data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)

    return pyg_data

# Dataset class
class OdorDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles_list = smiles_list
        self.labels = labels
        self.featurizer = GraphFeaturizer()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        Returns:
            tuple(Data, label): PyG Data object and corresponding label tensor
        """
        smiles = self.smiles_list[idx]
        try:
            graphdata = self.featurizer.featurize(smiles)
            if graphdata is None:
                raise ValueError(f"Graph featurization failed for SMILES: {smiles}")
            data = graphdata_to_pyg(graphdata)
        except Exception as e:
            print(f"Error featurizing SMILES at index {idx}: {e}")
            return None, None
        
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return data, label

# Custom collate function for PyTorch Geometric data
def collate_fn(batch):
    filtered_batch = [(data, label) for data, label in batch if data is not None and label is not None]
    
    if not filtered_batch:
        raise ValueError("Batch contains no valid samples after filtering.")

    graphs, labels = zip(*filtered_batch)
    batched_graph = Batch.from_data_list(graphs)
    labels = torch.stack(labels)

    return batched_graph, labels

# Molecule Feature Batching
class MoleculeDataBatch(Batch):
    @staticmethod
    def from_data_list(data_list):
        batch = Batch.from_data_list(data_list)
        
        # Handle molecular features separately
        mol_feats = torch.stack([d.mol_features for d in data_list])
        batch.mol_features = mol_feats
        
        return batch