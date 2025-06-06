from rdkit import Chem

# Define SMARTS patterns for functional groups
FG_SMARTS_PATTERNS = {
    'CarboxylicAcid': Chem.MolFromSmarts('C(=O)[O,H]'),
    'Alcohol': Chem.MolFromSmarts('[#6][OH]'),
    'Amine': Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),
    'Methoxy': Chem.MolFromSmarts('CO'),
    'Aldehyde': Chem.MolFromSmarts('[CX3H1](=O)[#6]'),
    'Acetyl': Chem.MolFromSmarts('CC(=O)'),
    'Nitrile': Chem.MolFromSmarts('C#N'),
    'Tert-butyl': Chem.MolFromSmarts('C(C)(C)C'),
    'Thiol': Chem.MolFromSmarts('[#6][SH]'),
    'Thioether': Chem.MolFromSmarts('[#6][S][#6]'),
    'Carbonyl': Chem.MolFromSmarts('C=O'),
    'Ethoxy': Chem.MolFromSmarts('CCO'),
    'Ester': Chem.MolFromSmarts('C(=O)O'),
    'Terpenes': Chem.MolFromSmarts('C=C(C)C'),  # General isoprene motif
    'Halogen': Chem.MolFromSmarts('[F,Cl,Br,I]')
}

FG_NAMES = list(FG_SMARTS_PATTERNS.keys())

def count_functional_groups(mol):
    """
    Returns a list with counts of each functional group in FG_NAMES.
    """
    counts = []
    for name in FG_NAMES:
        pattern = FG_SMARTS_PATTERNS[name]
        matches = mol.GetSubstructMatches(pattern)
        counts.append(len(matches))
    return counts
