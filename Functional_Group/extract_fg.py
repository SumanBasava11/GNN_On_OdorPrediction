import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import FunctionalGroups
from collections import Counter
from rdkit.Chem import Draw

# Load SMILES dataset
df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/OdorSmiles_Updated.csv', encoding='ISO-8859-1')

# Load RDKit functional group library
fgs = FunctionalGroups.BuildFuncGroupHierarchy()

# Prepare output file
output_file = 'Functional_Group/functional_groups_output.txt'

with_fg_count = 0
no_fg_count = 0
fg_counter = Counter()

with open(output_file, 'w') as f_out:
    # Write header
    f_out.write("SMILES\tFunctionalGroups\n")

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            f_out.write(f"{smiles}\tINVALID_MOLECULE\n")
            no_fg_count += 1
            continue

        matched_fgs = []
        for fg in fgs:
            if mol.HasSubstructMatch(fg.pattern):
                matched_fgs.append(fg.label)
                fg_counter[fg.label] += 1

        if matched_fgs:
            matched_str = ", ".join(matched_fgs)
            f_out.write(f"{smiles}\t{matched_str}\n")
            with_fg_count += 1
        else:
            no_fg_count += 1

print(f"Total molecules: {len(df)}")
print(f"Molecules with at least one functional group: {with_fg_count}")
print(f"Molecules with no functional group: {no_fg_count}")

# === Print functional group counts to the console ===
print("\nFunctional Group Counts:")
for fg, count in fg_counter.items():
    print(f"{fg}: {count}")


