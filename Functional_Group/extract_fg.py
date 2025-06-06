import os
import pandas as pd
from rdkit import Chem
from collections import Counter

# Define SMARTS patterns for custom functional groups
custom_fg_smarts = {
    "NC(=O)CH3": "[N;D2]-[C;D3](=O)-[C;D1;H3]",
    "-C(=O)O": "C(=O)[O;D1]",
    "-C(=O)OR": "C(=O)O[C;!$(C=O)]",
    "-C(=O)H": "C(=O)-[C;D1]",
    "-C(=O)N": "C(=O)-[N;D1]",
    "-C(=O)CH3": "C(=O)-[C;D1;H3]",
    "-N=C=O": "[N;D2]=[C;D2]=[O;D1]",
    "-N=C=S": "[N;D2]=[C;D2]=[S;D1]",
    "-NO2": "[N;D3](=[O;D1])[O;D1]",
    "-N=O": "[N;R0]=[O;D1]",
    "=N-O": "[N;R0]=[O;D1]",
    "=NCH3": "[N;R0]=[C;D1;H3]",
    "-N=CH2": "[N;R0]=[C;D1;H2]",
    "-N=NCH3": "[N;D2]=[N;D2]-[C;D1;H3]",
    "-N=N": "[N;D2]=[N;D1]",
    "-N#N": "[N;D2]#[N;D1]",
    "-C#N": "[C;D2]#[N;D1]",
    "-SO2NH2": "[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
    "-NHSO2CH3": "[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
    "-SO3H": "[S;D4](=O)(=O)-[O;D1]",
    "-SO3CH3": "[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
    "-SO2CH3": "[S;D4](=O)(=O)-[C;D1;H3]",
    "-SO2Cl": "[S;D4](=O)(=O)-[Cl]",
    "-SOCH3": "[S;D3](=O)-[C;D1]",
    "-SCH3": "[S;D2]-[C;D1;H3]",
    "-S": "[S;D1]",
    "=S": "[S;D1]",
    "-tBu": "[C;D4]([C;D1])([C;D1])-[C;D1]",
    "-C#CH": "[C;D2]#[C;D1;H]",
    "-cPropyl": "[C;D3]1-[C;D2]-[C;D2]1",
    "-OEt": "[O;D2]-[C;D2]-[C;D1;H3]",
    "-OMe": "[O;D2]-[C;D1;H3]",
    "-O": "[OX2H]",
    "=O": "[CX3]=O",
    "-N": "[N;X3;!$(N=*)]",
    "=N": "[N;X2]=C"
}

# Map labels to common names
label_to_common_name = {
    "NC(=O)CH3": "Acetamide",
    "-C(=O)O": "Carboxylic Acid",
    "-C(=O)OR": "Ester",
    "-C(=O)H": "Aldehyde",
    "-C(=O)N": "Amide",
    "-C(=O)CH3": "Acetyl",
    "-N=C=O": "Isocyanate",
    "-N=C=S": "Isothiocyanate",
    "-NO2": "Nitro",
    "-N=O": "Nitroso",
    "=N-O": "Nitroso Ether",
    "=NCH3": "Methyl Imino",
    "-N=CH2": "Iminomethyl",
    "-N=NCH3": "Methyl Azo",
    "-N=N": "Azo",
    "-N#N": "Azide",
    "-C#N": "Nitrile",
    "-SO2NH2": "Sulfonamide",
    "-NHSO2CH3": "Methylsulfonamide",
    "-SO3H": "Sulfonic Acid",
    "-SO3CH3": "Methyl Sulfonate",
    "-SO2CH3": "Methyl Sulfone",
    "-SO2Cl": "Sulfonyl Chloride",
    "-SOCH3": "Sulfoxide",
    "-SCH3": "Thioether",
    "-S": "Thiol",
    "=S": "Thione",
    "-tBu": "Tert-butyl",
    "-C#CH": "Terminal Alkyne",
    "-cPropyl": "Cyclopropyl",
    "-OEt": "Ethoxy",
    "-OMe": "Methoxy",
    "-O": "Alcohol",
    "=O": "Carbonyl",
    "-N": "Amine",
    "=N": "Imino"
}

# Compile SMARTS patterns
compiled_fg_patterns = {
    label: Chem.MolFromSmarts(smarts)
    for label, smarts in custom_fg_smarts.items()
     if smarts.strip() != "" and Chem.MolFromSmarts(smarts) is not None
}

# Load SMILES dataset
df = pd.read_csv(
    'C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/(Saturated)SoS_Full.csv',
    encoding='ISO-8859-1'
)

output_file = 'Functional_Group/functional_groups.txt'
with_fg_count = 0
no_fg_count = 0
fg_counter = Counter()

with open(output_file, 'w') as f_out:
    f_out.write("SMILES\tFunctionalGroups\n")

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            f_out.write(f"{smiles}\tINVALID_MOLECULE\n")
            no_fg_count += 1
            continue

        matched_fgs = []
        for label, pattern in compiled_fg_patterns.items():
            if mol.HasSubstructMatch(pattern):
                common_name = label_to_common_name.get(label, label)
                matched_fgs.append(common_name)
                fg_counter[common_name] += 1

        if matched_fgs:
            matched_str = ", ".join(matched_fgs)
            f_out.write(f"{smiles}\t{matched_str}\n")
            with_fg_count += 1
        else:
            f_out.write(f"{smiles}\tNO_MATCH\n")
            no_fg_count += 1

print(f"Total molecules: {len(df)}")
print(f"Molecules with at least one functional group: {with_fg_count}")
print(f"Molecules with no functional group: {no_fg_count}\n")

print("Functional Group Counts (by common name):")
for fg, count in fg_counter.items():
    print(f"{fg}: {count}")
