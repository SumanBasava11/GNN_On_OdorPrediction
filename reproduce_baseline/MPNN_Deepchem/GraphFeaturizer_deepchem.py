import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Data
from typing import List, Union, Dict, Sequence
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from Functional_Group.hard_encode_fgs import count_functional_groups
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_onehot(value, allowable_set: List, include_unknown_set=True) -> List[float]:
    return one_hot_encode(value, allowable_set, include_unknown_set)

def get_atomic_num_one_hot(atom: RDKitAtom,
                           allowable_set: List[int],
                           include_unknown_set: bool = True) -> List[float]:
    return one_hot_encode(atom.GetAtomicNum() - 1, allowable_set,
                          include_unknown_set)


def get_atom_total_valence_one_hot(
        atom: RDKitAtom,
        allowable_set: List[int],
        include_unknown_set: bool = True) -> List[float]:
    return one_hot_encode(atom.GetTotalValence(), allowable_set,
                          include_unknown_set)

class GraphConvConstants(object):
    """
    A class for holding featurization parameters.
    """

    MAX_ATOMIC_NUM = 36
    ATOM_FEATURES: Dict[str, List[int]] = {
        'valence': [0, 1, 2, 3, 4, 5, 6],
        'degree': [0, 1, 2, 3, 4, 5],
        'num_Hs': [0, 1, 2, 3, 4],
        'formal_charge': [-1, -2, 1, 2, 0],
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
        'num_radical_electrons': [0, 1],
        'is_aromatic': [0, 1],
        'is_in_ring': [0, 1],
        'smallest_ring': list(range(0, 16)),
    }
    ATOM_FEATURES_HYBRIDIZATION: List[str] = [
        "SP", "SP2", "SP3", "SP3D", "SP3D2"
    ]
    ATOM_FEATURES_CHIRALITY: List[str] = [
        'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER', 'CHI_TETRAHEDRAL', 'CHI_ALLENE', 'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL', 'CHI_OCTAHEDRAL'
    ]
    ATOM_FEATURES_BOND_TYPE: List[str] = [
        'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'
    ]
    # Dimension of atom feature vector
    ATOM_FDIM = sum(len(choices)
                    for choices in ATOM_FEATURES.values()) + len(
                        ATOM_FEATURES_HYBRIDIZATION) + len(ATOM_FEATURES_CHIRALITY) + len(ATOM_FEATURES_BOND_TYPE)
    # len(choices) +1 and len(ATOM_FEATURES_HYBRIDIZATION)
    # + 1 to include room for unknown set
    # BOND_FDIM = 6


def atom_features(atom: RDKitAtom) -> Sequence[Union[bool, int, float]]:
    """
    Compute atom features
    """
    if atom is None:
        features: Sequence[Union[bool, int,
                                 float]] = [0] * GraphConvConstants.ATOM_FDIM

    else:
        mol = atom.GetOwningMol()
        ring_info = mol.GetRingInfo()
        atom_idx = atom.GetIdx()

        # Smallest ring size
        ring_sizes = [len(r) for r in ring_info.AtomRings() if atom_idx in r]
        smallest_ring = min(ring_sizes) if ring_sizes else 0

        # Bonds connected
        bond_types_connected = set()
        for b in atom.GetBonds():
            bond_types_connected.add(str(b.GetBondType()))
        # encode the first one found or 'SINGLE' fallback
        first_bond_type = next(iter(bond_types_connected), 'SINGLE')

        features = []
        features += get_atom_total_valence_one_hot(atom, GraphConvConstants.ATOM_FEATURES['valence'])
        features += get_atom_total_degree_one_hot(atom, GraphConvConstants.ATOM_FEATURES['degree'])
        features += get_atom_total_num_Hs_one_hot(atom, GraphConvConstants.ATOM_FEATURES['num_Hs'])
        features += get_atom_formal_charge_one_hot(atom, GraphConvConstants.ATOM_FEATURES['formal_charge'])
        features += get_atomic_num_one_hot(atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
        
        # Radical electrons - encode using one_hot_encode from DeepChem
        features += one_hot_encode(
            atom.GetNumRadicalElectrons(),
            GraphConvConstants.ATOM_FEATURES['num_radical_electrons'],
            include_unknown_set=True)

        # Aromaticity (bool 0 or 1) encoded as one-hot
        features += one_hot_encode(
            int(atom.GetIsAromatic()),
            GraphConvConstants.ATOM_FEATURES['is_aromatic'],
            include_unknown_set=True)

        # Ring membership (bool 0 or 1) encoded as one-hot
        features += one_hot_encode(
            int(atom.IsInRing()),
            GraphConvConstants.ATOM_FEATURES['is_in_ring'],
            include_unknown_set=True)

        # Smallest ring size encoded as one-hot
        features += one_hot_encode(
            smallest_ring,
            GraphConvConstants.ATOM_FEATURES['smallest_ring'],
            include_unknown_set=True)
        
        features += get_onehot(str(atom.GetChiralTag()), GraphConvConstants.ATOM_FEATURES_CHIRALITY, True)
        features += get_atom_hybridization_one_hot(atom, GraphConvConstants.ATOM_FEATURES_HYBRIDIZATION, True)
        features += get_onehot(first_bond_type, GraphConvConstants. ATOM_FEATURES_BOND_TYPE, True)
        
        features = [int(feature) for feature in features]
    return features


def bond_features(bond: RDKitBond) -> Sequence[Union[bool, int, float]]:
    """
    Helper method used to compute bond feature vector.

    Parameters
    ----------
    bond: RDKitBond
        Bond to compute features on.

    Returns
    -------
    features: Sequence[Union[bool, int, float]]
        A list of bond features.
    """
    if bond is None:
        b_features: Sequence[Union[
            bool, int, float]] = [1] + [0] * (GraphConvConstants.BOND_FDIM - 1)

    else:
        bt = bond.GetBondType()
        b_features = [
            0, bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.IsInRing()
        ]

    return b_features

def molecular_features(mol: RDKitMol, mol_map: Dict[str, List]) -> np.ndarray:

    """
    Compute molecular features for a given RDKit Mol object.

    Scalar features are kept as floats.
    Categorical features are one-hot encoded based on mol_map.

    Parameters:
        mol: RDKitMol - RDKit molecule object
        mol_map: dict - dictionary mapping feature names to allowable sets for one-hot encoding
    
    Returns:
        np.ndarray: concatenated molecular feature vector
    """
    mol_features = []

    # def encode_feature(value, allowable_set):
    #     if len(allowable_set) == 1 and allowable_set[0] == 0:
    #         return [float(value)]
    #     else:
    #         return one_hot_encode(value, allowable_set, include_unknown_set=True)

    # molecular_weight
    mw = Descriptors.MolWt(mol)
    mol_features.append(float(mw))

    # logP (float)
    logp = Descriptors.MolLogP(mol)
    mol_features.append(float(logp))

    # TPSA (float)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    mol_features.append(float(tpsa))

    # num_rings (int categorical, keep as int)
    num_rings = mol.GetRingInfo().NumRings()
    mol_features.append(int(num_rings))

    # num_rotatable_bonds (int)
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    mol_features.append(int(num_rotatable_bonds))

    # num_H_bond_donors (int)
    num_H_donors = rdMolDescriptors.CalcNumHBD(mol)
    mol_features.append(int(num_H_donors))

    # num_H_bond_acceptors (int)
    num_H_acceptors = rdMolDescriptors.CalcNumHBA(mol)
    mol_features.append(int(num_H_acceptors))

    # heavy_atom_count (int)
    heavy_atom_count = rdMolDescriptors.CalcNumHeavyAtoms(mol)
    mol_features.append(int(heavy_atom_count))

    # formal_charge (int)
    formal_charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    mol_features.append(int(formal_charge))

    # complexity (float)
    complexity = Descriptors.BertzCT(mol)
    mol_features.append(float(complexity))

    # Functional group counts
    fg_counts = count_functional_groups(mol)
    mol_features.extend(fg_counts)

    return np.array(mol_features, dtype=np.float32)

class GraphFeaturizer(MolecularFeaturizer):

    def __init__(self, is_adding_hs=False):

        self.is_adding_hs = is_adding_hs
        super(GraphFeaturizer).__init__()

        # Molecular feature map for external function
        self.mol_map = {
            'molecular_weight': [0],
            'logp': [0],
            'tpsa': [0],
            'num_rings': list(range(0, 38)),
            'num_rotatable_bonds': list(range(0, 149)),
            'num_H_bond_donors': list(range(0, 116)),
            'num_H_bond_acceptors': list(range(0, 191)),
            'heavy_atom_count': list(range(0, 419)),
            'formal_charge': list(range(-2, 2)),
            'complexity': [0],
        }


    def _construct_bond_index(self, datapoint: RDKitMol) -> np.ndarray:
        
        src: List[int] = []
        dest: List[int] = []
        for bond in datapoint.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]
        return np.asarray([src, dest], dtype=int)

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        if isinstance(datapoint, Chem.rdchem.Mol):
            if self.is_adding_hs:
                datapoint = Chem.AddHs(datapoint)
        else:
            raise ValueError(
                "Feature field should contain smiles for featurizer!")

        # get atom features
        f_atoms: np.ndarray = np.asarray(
            [atom_features(atom) for atom in datapoint.GetAtoms()],
            dtype=float)

        # # get edge(bond) features
        # if len(datapoint.GetBonds()) == 0:
        #     f_bonds: np.ndarray = np.empty((0, GraphConvConstants.BOND_FDIM))
        # else:
        #     f_bonds_list = []
        #     for bond in datapoint.GetBonds():
        #         b_feat = 2 * [bond_features(bond)]
        #         f_bonds_list.extend(b_feat)
        #     f_bonds = np.asarray(f_bonds_list, dtype=float)

        # get edge index
        edge_index: np.ndarray = self._construct_bond_index(datapoint)
        mol_features = molecular_features(datapoint, self.mol_map)

        return GraphData(node_features=f_atoms,
                         edge_index=edge_index,
                         mol_features=mol_features)     #edge_features=f_bonds

    def featurize(self, smiles: str) -> Data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        graph_data = self._featurize(mol)

        x = torch.tensor(graph_data.node_features, dtype=torch.float)
        edge_index = torch.tensor(graph_data.edge_index, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)

        # Optional molecular-level features if available
        if hasattr(graph_data, 'mol_features'):
            data.mol_features = torch.tensor(graph_data.mol_features, dtype=torch.float)

        return data
    
def main():
    # Load CSV file with SMILES strings
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv', encoding='ISO-8859-1')
     
    featurizer = GraphFeaturizer(is_adding_hs=False)
    
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES at index {idx}: {smiles}")
            continue
        
        graph_data = featurizer._featurize(mol)
        node_features = graph_data.node_features
        mol_features = molecular_features(mol, featurizer.mol_map)
        
        print(f"\nMolecule index: {idx}, SMILES: {smiles}")
        print(f"Number of atoms: {node_features.shape[0]}")
        print(f"Dimension of each atom feature vector: {node_features.shape[1]}")
        print("Atom features:")
        for i, features in enumerate(node_features):
            print(f"  Atom {i}: {features}")

        print("Molecular features:")
        print(mol_features)
        print(f"Dimension of molecular feature vector: {len(mol_features)}")

if __name__ == "__main__":
    main()