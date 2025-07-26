from rdkit import Chem
import torch
from Featurizer.feature_maps import x_map

def get_node_features(mol):
    node_features = []

    # Column descriptions for regular features
    column_descriptions = [
        'atomic_num', 'degree', 'formal_charge', 'num_hs', 'num_radical_electrons', 
        'valence', 'is_aromatic', 'is_in_ring', 'smallest_ring', 'chirality', 
        'hybridization'
    ]
    
    # # For bond types, directly use the bond types from x_map
    bond_types_columns = x_map['bond_types_connected']

    # Combine all column descriptions (regular + bond types)
    column_descriptions.extend(bond_types_columns)

    for atom in mol.GetAtoms():
        # Get all atom properties
        atomic_num = atom.GetAtomicNum()
        valence = atom.GetTotalValence()
        degree = atom.GetTotalDegree()
        num_hs = atom.GetTotalNumHs()
        num_radical_electrons = atom.GetNumRadicalElectrons()
        formal_charge = atom.GetFormalCharge()

        # Safe access for 'is_aromatic' and 'is_in_ring' boolean features
        is_aromatic = x_map['is_aromatic'].get(atom.GetIsAromatic(), -1)
        in_ring = x_map['is_in_ring'].get(atom.IsInRing(), -1)

        # Smallest ring size
        ring_sizes = [r for r in range(3, 9) if atom.IsInRingSize(r)]
        smallest_ring = min(ring_sizes) if ring_sizes else 0

        # Safe categorical encodings for chirality and hybridization
        chirality = x_map['chirality'].get(str(atom.GetChiralTag()), -1)
        hybridization = x_map['hybridization'].get(str(atom.GetHybridization()), -1)

        # Multi-hot bond types
        bond_types = {str(b.GetBondType()) for b in atom.GetBonds()}
        bond_types_mh = [1 if bt in bond_types else 0 for bt in bond_types_columns]

        # Constructing the feature vector
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
            chirality,
            hybridization
        ] + bond_types_mh

        # Append the feature vector to the node_features list
        node_features.append(features)

    # Convert to tensor
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    return node_features_tensor
