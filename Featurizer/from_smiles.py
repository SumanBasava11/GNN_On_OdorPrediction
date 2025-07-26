from rdkit import Chem
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from Featurizer.node_features import get_node_features
from Featurizer.edge_features import get_edge_features
from Featurizer.mol_features import *

def from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[Invalid SMILES] {smiles}")
        return None
    try:
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
            smiles=smiles,
            mol_features = mol_feat
        )

        return data
    
    except Exception as e:
        print(f"[from_smiles ERROR] {smiles} => {e}")
        return None
        