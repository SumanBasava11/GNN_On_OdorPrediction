import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, MolFromSmarts
from Featurizer.feature_maps import mol_map
from Functional_Group.hard_encode_fgs import count_functional_groups

def longest_carbon_chain(mol):
    """
    Returns the length of the longest linear carbon chain in the molecule.
    """
    def dfs(atom, visited, length):
        visited[atom.GetIdx()] = True
        max_len = length
        for neighbor in atom.GetNeighbors():
            if not visited[neighbor.GetIdx()] and neighbor.GetAtomicNum() == 6:
                max_len = max(max_len, dfs(neighbor, visited, length + 1))
        visited[atom.GetIdx()] = False
        return max_len

    max_chain = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:  # Carbon
            visited = [False] * mol.GetNumAtoms()
            max_chain = max(max_chain, dfs(atom, visited, 1))

    return max_chain

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
        longest_carbon_chain(mol)
    ]

    fg_counts = count_functional_groups(mol)
    full_feats =  features + fg_counts
    return torch.tensor(full_feats, dtype=torch.float)