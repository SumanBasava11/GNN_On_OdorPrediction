import os
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from torch_geometric.data import Data
from Featurizer.from_smiles import from_smiles  # Make sure this works in your context

# Read the CSV file
df = pd.read_csv(
    'C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/Balanced_OdorSmiles_Top100.csv',
    encoding='ISO-8859-1'
)

all_node_features = []
all_mol_features = []

# Loop through each SMILES and collect features
for idx, row in df.iterrows():
    smiles = row['SMILES']
    data = from_smiles(smiles)
    if data is None:
        continue

    try:
        all_node_features.append(data.x.cpu())  # Shape: [num_atoms, node_feat_dim]
        all_mol_features.append(torch.tensor(data.mol_features).view(1, -1))  # Shape: [1, mol_feat_dim]
    except Exception as e:
        print(f"[ERROR] Processing {smiles}: {e}")

# Stack features into matrices
all_node_features = torch.cat(all_node_features, dim=0)  # [total_atoms, num_node_features]
all_mol_features = torch.cat(all_mol_features, dim=0)    # [num_molecules, num_mol_features]

node_features_np = all_node_features.numpy()
mol_features_np = all_mol_features.numpy()

# Create output directory
os.makedirs("plots/feature_distributions", exist_ok=True)

# Helper function to plot histograms
def plot_feature_distributions(features_np, feature_type="node"):
    
    output_dir = f"plots/feature_distributions/{feature_type}"
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    num_features = features_np.shape[1]
    for i in range(num_features):
        plt.figure(figsize=(6, 4))
        sns.histplot(features_np[:, i], bins=50, kde=True, color="skyblue")
        plt.title(f"{feature_type.capitalize()} Feature {i} Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"plots/feature_distributions/{feature_type}_feature_{i}.png")
        plt.close()

# Plot and save the distributions
node_feature_names = [f"NodeFeat_{i}" for i in range(node_features_np.shape[1])]
mol_feature_names = [f"MolFeat_{i}" for i in range(mol_features_np.shape[1])]

print(f"Saved feature distribution plots to: plots/feature_distributions/")
