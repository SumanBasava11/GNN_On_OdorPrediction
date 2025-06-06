import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from tqdm import tqdm

from Featurizer.mol_features import *
from Functional_Group.hard_encode_fgs import *

def plot_and_save(feature_df, feature_names, continuous_features, discrete_features, output_dir, plots_per_figure=12):
    os.makedirs(output_dir, exist_ok=True)

    all_features = continuous_features + discrete_features
    total_features = len(all_features)
    num_figures = int(np.ceil(total_features / plots_per_figure))

    for fig_num in range(num_figures):
        start = fig_num * plots_per_figure
        end = min((fig_num + 1) * plots_per_figure, total_features)
        selected_features = all_features[start:end]

        cols = 4
        rows = int(np.ceil(len(selected_features) / cols))
        plt.figure(figsize=(5.5 * cols, 3.5 * rows))

        for i, feature in enumerate(selected_features):
            plt.subplot(rows, cols, i + 1)
            values = feature_df[feature].dropna()

            if feature in continuous_features:
                sns.kdeplot(values, fill=True, bw_adjust=1.2)
                plt.axvline(values.min(), color='red', linestyle='--')
                plt.axvline(values.max(), color='green', linestyle='--')
                plt.text(values.min(), plt.ylim()[1] * 0.9, f"min: {values.min():.2f}", color='red', fontsize=8, ha='left')
                plt.text(values.max(), plt.ylim()[1] * 0.9, f"max: {values.max():.2f}", color='green', fontsize=8, ha='right')
                plt.title(feature, fontsize=10)
                plt.xlabel(feature, fontsize=8)
                plt.ylabel("Density", fontsize=8)
            else:
                values = values.astype(int)
                sns.boxplot(x=values, color="skyblue", orient="h", fliersize=3, linewidth=1)
                plt.title(feature, fontsize=10)
                plt.xlabel(feature, fontsize=8)
                plt.yticks([])

                plt.axvline(values.min(), color='red', linestyle='--')
                plt.axvline(values.max(), color='green', linestyle='--')
                plt.text(values.min(), 0.8, f"min: {values.min()}", color='red', fontsize=7, ha='left')
                plt.text(values.max(), 0.8, f"max: {values.max()}", color='green', fontsize=7, ha='right')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_plot_part_{fig_num+1}.png", dpi=300)
        plt.close()

def main():
    # Load SMILES data
    df = pd.read_csv(
        "C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/(Saturated)SoS_Full.csv",
        encoding='ISO-8859-1'
    )
    smiles_list = df["SMILES"].dropna().tolist()

    # Extract features
    feature_data = []
    valid_smiles = []

    for smi in tqdm(smiles_list, desc="Extracting features"):
        mol = Chem.MolFromSmiles(smi)
        feats = get_molecular_features(mol)
        if feats is not None:
            feature_data.append(feats.tolist())
            valid_smiles.append(smi)

    # Define full feature names
    feature_names = [
        "molecular_weight", "logp", "tpsa", "num_rings", "num_rotatable_bonds",
        "num_H_bond_donors", "num_H_bond_acceptors", "heavy_atom_count", 
        "formal_charge", "FractionCSP3", "longest_carbon_chain"
    ] + FG_NAMES

    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_data, columns=feature_names)

    # Separate feature types
    continuous_features = [
        "molecular_weight", "logp", "tpsa", "FractionCSP3"
    ]
    discrete_features = [f for f in feature_names if f not in continuous_features]

    # Plot and save
    plot_and_save(feature_df, feature_names, continuous_features, discrete_features, output_dir="Featurizer/mol_feature_plots")

if __name__ == "__main__":
    main()
