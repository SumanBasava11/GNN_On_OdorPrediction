from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/(Saturated)SoS_Full.csv', encoding='ISO-8859-1')

cas_col = 'cas_number'
smiles_col = 'SMILES'

# List to hold results
results = []

# Iterate through each row
for idx, row in df.iterrows():
    smiles = row[smiles_col]
    cas_number = row[cas_col]
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue
    
    mol_wt = Descriptors.MolWt(mol)
    
    if mol_wt > 500:
        results.append((cas_number, smiles, mol_wt))

# Save results to a TXT file
output_path = 'Featurizer/high_mol_weight_molecules.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("CAS_Number\tSMILES\tMolecular_Weight\n")
    for cas, smi, mw in results:
        f.write(f"{cas}\t{smi}\t{mw:.2f}\n")

print(f"Saved {len(results)} molecules with MW > 500 to {output_path}")