import torch
from rdkit.Chem import Descriptors, Lipinski
from Featurizer.feature_maps import mol_map

def get_molecular_features(mol):
    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        len(mol.GetRingInfo().AtomRings()),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.HeavyAtomCount(mol),
        sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        Descriptors.FractionCSP3(mol),
    ]
    return torch.tensor(features, dtype=torch.float)
