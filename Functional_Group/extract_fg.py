import os
import pandas as pd
from rdkit import Chem
from collections import Counter

# Define SMARTS patterns for requested functional groups
fg_smarts = {
    "Acid": "[CX3](=O)[OX2H1]",                  # Carboxylic Acid group -COOH
    "Acetamide": "NC(=O)C",                      # Acetamide group
    "Alcohols": "[OX2H]",                        # Alcohol OH
    "Acetyl": "CC(=O)[#6]",                      # Acetyl group CH3-CO-
    "Aldehydes": "[CX3H1](=O)[#6]",             # Aldehyde group -CHO
    "Alkanes": "[CX4]",                          # Alkane carbon (sp3)
    "Amide": "C(=O)N",                           # Amide group
    "Amine": "[NX3;H2,H1;!$(NC=O)]",             # Primary or secondary amines not attached to carbonyl
    "Azide": "[N-]=[N+]=N",                      # Azide group
    "Bicyclic": "[$([R2]),$([R3])]",             # Atoms in 2 or 3 rings (approx for bicyclic)
    "Cyclic": "[$([R])]",                        # Any ring atom
    "Carbonyl": "[CX3]=O",                       # Carbonyl group C=O
    "Esters": "C(=O)O[C]",                       # Ester group
    "Carboxylic Acid": "[CX3](=O)[OX2H1]",      # Carboxylic Acid group -COOH (same as Acid)
    "Ethers": "[OD2]([#6])[#6]",                 # Ether -O-
    "Cyclopropyl": "C1CC1",                      # Cyclopropyl ring
    "Furan": "c1ccoc1",                          # Furan ring
    "Hydrocarbons": "[CX4H]",                    # Hydrocarbon alkane carbon with Hs
    "Ethoxy": "OCC",                             # Ethoxy group -OCH2CH3
    "Isothiocyanates": "N=C=S",                  # Isothiocyanate group
    "Imino": "[NX2]=C",                          # Imino group
    "Ketones": "C(=O)[#6]",                      # Ketone group
    "Lactone": "O=C1OC=CC1",                     # Generic lactone ring (simple approximation)
    "N-Compounds": "[NX3]",                       # Any nitrogen with three connections
    "Isocyanate": "N=C=O",                       # Isocyanate group
    "Oximes": "C=NO",                            # Oxime group (C=NOH)
    "Methoxy": "OC",                             # Methoxy group -OCH3
    "Oxirane": "C1OC1",                          # Epoxide (oxirane) ring
    "Phenol": "c1ccc(cc1)O",                     # Phenol group
    "Pyran": "c1ccoc1",                          # Pyran ring (similar to furan with O)
    "Pyrazine": "c1cnccn1",                      # Pyrazine ring
    "Pyrrole": "c1cc[nH]c1",                     # Pyrrole ring
    "Pyridine": "c1ccncc1",                      # Pyridine ring
    "S-compounds": "[SX2]",                       # Sulfur compounds (S with 2 connections)
    "Spiro": "[*]1[*]2[*]1[*]2",                 # Approximate spiro rings (very rough)
    "Sulfides": "CSC",                           # Thioether (sulfide) -C-S-C-
    "Thiazoles": "c1cscn1",                       # Thiazole ring
    "Nitro": "[N+](=O)[O-]",                     # Nitro group
    "Sulfonamide": "S(=O)(=O)N",                 # Sulfonamide group
    "Thioesters": "C(=O)S",                      # Thioester group
    "Thiols": "[SX2H]",                          # Thiol group (-SH)
    "Halogen": "[F,Cl,Br,I]",                     # Any halogen atom
    "Tert-butyl": "C(C)(C)C",                    # Tert-butyl group
    "Nitrile": "C#N",                            # Nitrile group
}

# Compile SMARTS patterns
compiled_patterns = {}
for name, smarts in fg_smarts.items():
    mol = Chem.MolFromSmarts(smarts)
    if mol:
        compiled_patterns[name] = mol
    else:
        print(f"Warning: SMARTS pattern failed to compile for {name}: {smarts}")

# Load your SMILES dataset
df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv', encoding='ISO-8859-1')  # Change path and filename as needed

output_path = 'functional_groups_output.txt'

fg_counter = Counter()
total_with_fg = 0
total_no_fg = 0

with open(output_path, 'w') as out_file:
    out_file.write("SMILES\tFunctionalGroups\n")

    for idx, row in df.iterrows():
        smiles = row['SMILES']  # Make sure your CSV has a 'SMILES' column
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            out_file.write(f"{smiles}\tINVALID_MOLECULE\n")
            total_no_fg += 1
            continue

        matched = []
        for fg_name, pattern in compiled_patterns.items():
            if mol.HasSubstructMatch(pattern):
                matched.append(fg_name)
                fg_counter[fg_name] += 1

        if matched:
            total_with_fg += 1
            out_file.write(f"{smiles}\t{', '.join(matched)}\n")
        else:
            total_no_fg += 1
            out_file.write(f"{smiles}\tNO_MATCH\n")

print(f"Total molecules: {len(df)}")
print(f"With functional groups: {total_with_fg}")
print(f"No functional groups: {total_no_fg}\n")

print("Functional group counts:")
for fg, count in fg_counter.most_common():
    print(f"{fg}: {count}")
