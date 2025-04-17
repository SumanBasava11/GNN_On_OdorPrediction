from rdkit import Chem
import torch
from Featurizer.feature_maps import x_map

def get_node_features(mol):
    node_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        valence = atom.GetTotalValence()
        degree = atom.GetTotalDegree()
        num_hs = atom.GetTotalNumHs()
        num_radical_electrons = atom.GetNumRadicalElectrons()
        formal_charge = atom.GetFormalCharge()
        is_aromatic = atom.GetIsAromatic()
        in_ring = atom.IsInRing()

        # Smallest ring size
        ring_sizes = [r for r in range(3, 9) if atom.IsInRingSize(r)]
        smallest_ring = min(ring_sizes) if ring_sizes else 0

        # One-hot categorical encodings
        chirality = str(atom.GetChiralTag())
        hybridization = str(atom.GetHybridization())
        bond_types_connected = {str(b.GetBondType()) for b in atom.GetBonds()}

        chirality_oh = [1 if chirality == c else 0 for c in x_map['chirality']]
        hybrid_oh = [1 if hybridization == h else 0 for h in x_map['hybridization']]
        bond_types_oh = [1 if b in bond_types_connected else 0 for b in x_map['bond_types_connected']]

        features = [
            atomic_num,
            degree,
            formal_charge,
            num_hs,
            num_radical_electrons,
            valence,
            is_aromatic,
            in_ring,
            smallest_ring,
        ] + chirality_oh + hybrid_oh + bond_types_oh

        node_features.append(features)

    return torch.tensor(node_features, dtype=torch.float)
