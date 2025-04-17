from rdkit import Chem
from torch_geometric.data import Data
from Featurizer.node_features import get_node_features
from Featurizer.edge_features import get_edge_features
from Featurizer.mol_features import get_molecular_features

def from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    node_feat = get_node_features(mol)
    edge_index, edge_attr = get_edge_features(mol, num_nodes=node_feat.size(0))
    mol_feat = get_molecular_features(mol)

    data = Data(
        x=node_feat, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        smiles=smiles
    )
    
    data.mol_features = mol_feat

    return data
