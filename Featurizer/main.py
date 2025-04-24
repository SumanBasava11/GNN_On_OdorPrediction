import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data

from Featurizer.feature_maps import x_map
from Featurizer.from_smiles import from_smiles
from Featurizer.node_features import get_node_features
from Featurizer.edge_features import get_edge_features
from Featurizer.mol_features import get_molecular_features


def main():

    # Load the CSV file containing SMILES strings
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/OdorSmiles_Updated.csv', encoding='ISO-8859-1')
    output_path = "Featurizer/smiles_features_output.txt"

    with open(output_path, "w") as f:
        for index, row in df.iterrows():
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            try:
                x = get_node_features(mol)
                num_nodes = x.size(0)
                edge_index, edge_attr = get_edge_features(mol, num_nodes)
                mol_features = get_molecular_features(mol)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
                data.mol_features = mol_features

                f.write(f"\nFeatures for SMILES: {smiles}\n")
                f.write("-" * 60 + "\n")
                f.write("Node Feature Matrix:\n")
                f.write(str(data.x) + "\n\n")

                f.write("Molecular Features:\n")
                f.write(str(data.mol_features) + "\n\n")

                f.write("=" * 80 + "\n")

            except Exception as e:
                print(f"[ERROR] Failed on SMILES {smiles}: {e}")

    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()