# Node features
x_map = {
    'atomic_num': list(range(0, 35)),
    'degree': list(range(0, 4)),
    'formal_charge': list(range(-2, 2)),
    'num_hs': list(range(0, 5)),
    'num_radical_electrons': list(range(0, 1)),
    'valence': list(range(0, 6)),
    'is_aromatic': {False: 0, True: 1},
    'is_in_ring': {False: 0, True: 1},
    'smallest_ring': list(range(0, 15)),
    'chirality' : {
        'CHI_UNSPECIFIED': 0,
        'CHI_TETRAHEDRAL_CW': 1,
        'CHI_TETRAHEDRAL_CCW': 2,
        'CHI_OTHER': 3,
        'CHI_TETRAHEDRAL': 4,
        'CHI_ALLENE': 5,
        'CHI_SQUAREPLANAR': 6,
        'CHI_TRIGONALBIPYRAMIDAL': 7,
        'CHI_OCTAHEDRAL': 8
    },
    'hybridization' : {
        'UNSPECIFIED': 0,
        'S': 1,
        'SP': 2,
        'SP2': 3,
        'SP3': 4,
        'SP3D': 5,
        'SP3D2': 6,
        'OTHER': 7
    },
    'bond_types_connected' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC'
    ]
}

# Edge features
e_map = {
    'stereo' : ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'],
    'is_conjugated': [False, True],
    # 'bond_types_connected' : [
    #     'SINGLE',
    #     'DOUBLE',
    #     'TRIPLE',
    #     'AROMATIC'
    # ]
}

# Molecular features
mol_map = {
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
