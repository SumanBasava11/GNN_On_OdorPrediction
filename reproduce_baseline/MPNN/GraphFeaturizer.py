import torch
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from typing import List, Sequence, Union, Dict
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm

# Constants and feature dimensions container
class GraphConvConstants(object):
    """
    A class for holding featurization parameters.
    """
    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES: Dict[str, List[int]] = {
        'valence': [0, 1, 2, 3, 4, 5, 6],
        'degree': [0, 1, 2, 3, 4, 5],
        'num_Hs': [0, 1, 2, 3, 4],
        'formal_charge': [-2, -1, 0, 1, 2],
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
    }
    ATOM_FEATURES_HYBRIDIZATION: List[HybridizationType] = [
        HybridizationType.SP, 
        HybridizationType.SP2, 
        HybridizationType.SP3, 
        HybridizationType.SP3D, 
        HybridizationType.SP3D2
    ]
    # Dimension of atom feature vector = sum of all one-hot vectors (+1 for unknown) + hybridization + unknown hybridization
    ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + len(ATOM_FEATURES_HYBRIDIZATION) + 1
    BOND_FDIM = 6


def one_hot_encoding_with_unknown(value, allowable_set, include_unknown=True) -> List[int]:
    """
    One-hot encode 'value' based on allowable_set.
    If value not in set, encode as unknown (last position).
    """
    if value in allowable_set:
        return [int(value == s) for s in allowable_set] + ([0] if include_unknown else [])
    else:
        return [0] * len(allowable_set) + ([1] if include_unknown else [])


def get_atom_total_valence_one_hot(atom, allowable_set=None, include_unknown=True):
    if allowable_set is None:
        allowable_set = GraphConvConstants.ATOM_FEATURES['valence']
    valence = atom.GetTotalValence()
    return one_hot_encoding_with_unknown(valence, allowable_set, include_unknown)


def get_atom_total_degree_one_hot(atom, allowable_set=None, include_unknown=True):
    if allowable_set is None:
        allowable_set = GraphConvConstants.ATOM_FEATURES['degree']
    degree = atom.GetTotalDegree()
    return one_hot_encoding_with_unknown(degree, allowable_set, include_unknown)


def get_atom_total_num_Hs_one_hot(atom, allowable_set=None, include_unknown=True):
    if allowable_set is None:
        allowable_set = GraphConvConstants.ATOM_FEATURES['num_Hs']
    num_Hs = atom.GetTotalNumHs()
    return one_hot_encoding_with_unknown(num_Hs, allowable_set, include_unknown)


def get_atom_formal_charge_one_hot(atom, allowable_set=None, include_unknown=True):
    if allowable_set is None:
        allowable_set = GraphConvConstants.ATOM_FEATURES['formal_charge']
    formal_charge = atom.GetFormalCharge()
    return one_hot_encoding_with_unknown(formal_charge, allowable_set, include_unknown)


def get_atomic_num_one_hot(atom, allowable_set=None, include_unknown=True):
    if allowable_set is None:
        allowable_set = GraphConvConstants.ATOM_FEATURES['atomic_num']
    atomic_num = atom.GetAtomicNum()
    return one_hot_encoding_with_unknown(atomic_num, allowable_set, include_unknown)


def get_atom_hybridization_one_hot(atom, allowable_set=None, include_unknown=True):
    if allowable_set is None:
        allowable_set = GraphConvConstants.ATOM_FEATURES_HYBRIDIZATION
    hybridization = atom.GetHybridization()
    # The allowable_set is a list of HybridizationType enums
    if hybridization in allowable_set:
        one_hot = [int(hybridization == h) for h in allowable_set]
        if include_unknown:
            one_hot.append(0)
        return one_hot
    else:
        # unknown hybridization
        if include_unknown:
            return [0] * len(allowable_set) + [1]
        else:
            return [0] * len(allowable_set)


def atom_features(atom) -> List[int]:
    """
    Compute the full atom feature vector.
    """
    if atom is None:
        return [0] * GraphConvConstants.ATOM_FDIM

    features = []
    features += get_atom_total_valence_one_hot(atom)
    features += get_atom_total_degree_one_hot(atom)
    features += get_atom_total_num_Hs_one_hot(atom)
    features += get_atom_formal_charge_one_hot(atom)
    features += get_atomic_num_one_hot(atom)
    features += get_atom_hybridization_one_hot(atom)
    # Convert bool/int features explicitly to int
    features = [int(f) for f in features]
    return features


def bond_features(bond) -> List[int]:
    """
    Compute bond features vector.
    """
    if bond is None:
        # If no bond, encode as 'no bond' with 1 at first pos, zeros elsewhere (length 6)
        return [1] + [0] * (GraphConvConstants.BOND_FDIM - 1)

    bt = bond.GetBondType()
    return [
        0,
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.IsInRing()),
    ]


class GraphFeaturizerTorch:
    def __init__(self, add_hs=False):
        self.add_hs = add_hs
    
    def __call__(self, smiles: str):
        if isinstance(smiles, pd.Series):
            return [self.featurize(s) for s in smiles]
        else:
            return self.featurize(smiles)

    def _construct_bond_index(self, mol):
        src = []
        dst = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            # Add both directions
            src += [start, end]
            dst += [end, start]
        return torch.tensor([src, dst], dtype=torch.long)

    def featurize(self, smiles: str) -> Data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        if self.add_hs:
            mol = Chem.AddHs(mol)

        # Node features
        atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)

        # Edge features and edge indices
        edge_index = self._construct_bond_index(mol)
        if mol.GetNumBonds() == 0:
            edge_attr = torch.empty((0, GraphConvConstants.BOND_FDIM), dtype=torch.float)
        else:
            bond_feats = []
            for bond in mol.GetBonds():
                bf = bond_features(bond)
                # add both directions
                bond_feats += [bf, bf]
            edge_attr = torch.tensor(bond_feats, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# def main(csv_path: str, output_txt_path: str):
#     df = pd.read_csv(csv_path)
#     if 'smiles' not in df.columns:
#         raise ValueError("CSV must have a 'smiles' column.")
    
#     featurizer = GraphFeaturizerTorch(add_hs=False)
#     invalid_smiles = []

#     with open(output_txt_path, 'w') as f:
#         for idx, smiles in tqdm(enumerate(df['smiles']), total=len(df), desc="Featurizing molecules"):
#             try:
#                 data = featurizer.featurize(smiles)
#                 f.write(f"Features for SMILES: {smiles}\n")
#                 f.write("------------------------------------------------------------\n")
#                 f.write("Node Feature Matrix:\n")
#                 f.write(str(data.x) + "\n\n")
#                 print(data.x.shape)
                
#                 if data.edge_attr is not None and data.edge_attr.size(0) > 0:
#                     f.write("Edge Feature Matrix:\n")
#                     f.write(str(data.edge_attr) + "\n\n")
#                     print(data.edge_attr.shape)
#                 else:
#                     f.write("Edge Feature Matrix: None\n\n")
                
#                 f.write("Edge Index:\n")
#                 f.write(str(data.edge_index) + "\n\n")
                
#             except Exception as e:
#                 print(f"[Warning] Failed to process SMILES at index {idx}: {smiles} ({str(e)})")
#                 invalid_smiles.append((idx, smiles))

#     print(f"\nTotal molecules processed: {len(df) - len(invalid_smiles)}")
#     print(f"Invalid SMILES skipped: {len(invalid_smiles)}")

# if __name__ == "__main__":
#     csv_file_path = "C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv"  # your CSV path
#     output_file_path = "C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/featurized_molecules.txt"  # your output txt file path
#     main(csv_file_path, output_file_path)