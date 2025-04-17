import torch
from Featurizer.feature_maps import e_map

def get_edge_features(mol, num_nodes):
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        stereo = str(bond.GetStereo())
        stereo_idx = e_map['stereo'].index(stereo) if stereo in e_map['stereo'] else 0

        e = [
            stereo_idx,
            e_map['is_conjugated'].index(bond.GetIsConjugated()),
        ]

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    if edge_index.numel() > 0:
        perm = (edge_index[0] * num_nodes + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

    return edge_index, edge_attr
