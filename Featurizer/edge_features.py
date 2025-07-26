import torch
from Featurizer.feature_maps import e_map

def get_edge_features(mol, num_nodes):
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Edge feature 1: Bond stereo
        stereo = str(bond.GetStereo())
        stereo_idx = e_map['stereo'].index(stereo) if stereo in e_map['stereo'] else 0
        
        e = [
            stereo_idx,
            e_map['is_conjugated'].index(bond.GetIsConjugated()),
        ]

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

        # # Edge feature 2: Is conjugated
        # is_conjugated_idx = e_map['is_conjugated'].index(bond.GetIsConjugated())

        # # Edge feature 3: Bond type (SINGLE, DOUBLE, etc.)
        # bond_type = str(bond.GetBondType())
        # bond_type_idx = e_map['bond_types_connected'].index(bond_type) if bond_type in e_map['bond_types_connected'] else 0

        # e = [stereo_idx, is_conjugated_idx, bond_type_idx]

        # edge_indices += [[i, j], [j, i]]
        # edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    if edge_index.numel() > 0:
        perm = (edge_index[0] * num_nodes + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

    return edge_index, edge_attr
