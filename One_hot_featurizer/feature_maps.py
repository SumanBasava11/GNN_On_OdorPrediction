import torch
import torch.nn.functional as F

x_map = {
    'atomic_num': list(range(0, 35)),
    'degree': list(range(0, 4)),
    'formal_charge': list(range(-2, 2)),
    'num_hs': list(range(0, 5)),
    'num_radical_electrons': list(range(0, 1)),
    'valence': list(range(0, 6)),
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
    'smallest_ring': list(range(0, 15)),
    'chirality': [
        'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER', 'CHI_TETRAHEDRAL', 'CHI_ALLENE', 'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL', 'CHI_OCTAHEDRAL'
    ],
    'hybridization': [
        'UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'
    ],
    'bond_types_connected': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
}

e_map = {
    'stereo': ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'],
    'is_conjugated': [False, True],
}

mol_map = {
    'molecular_weight': [0],  # keep as float
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

def encode_onehot(value, choices):
    """
    Encode a categorical value as one-hot tensor
    """
    if isinstance(choices[0], bool):
        idx = int(value)
    else:
        idx = choices.index(value)
    return F.one_hot(torch.tensor(idx), num_classes=len(choices)).float()

def extract_node_features(node_data):
    """
    node_data: dict of raw node properties
    returns: concatenated one-hot feature tensor
    """
    feats = []
    for key, choices in x_map.items():
        val = node_data[key]
        onehot = encode_onehot(val, choices)
        feats.append(onehot)
    return torch.cat(feats)

def extract_edge_features(edge_data):
    """
    edge_data: dict of raw edge properties
    returns: concatenated one-hot feature tensor
    """
    feats = []
    for key, choices in e_map.items():
        val = edge_data[key]
        onehot = encode_onehot(val, choices)
        feats.append(onehot)
    return torch.cat(feats)


def extract_mol_features(mol_data):
    """
    mol_data: dict of raw molecular properties
    returns: concatenated feature tensor
    """
    feats = []
    for key, choices in mol_map.items():
        val = mol_data[key]
        if choices is None:
            # continuous value, keep as is
            feats.append(torch.tensor([val], dtype=torch.float))
        else:
            onehot = encode_onehot(val, choices)
            feats.append(onehot)
    return torch.cat(feats)