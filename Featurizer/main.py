import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data

from Featurizer.from_smiles import from_smiles
from Featurizer.node_features import get_node_features
from Featurizer.edge_features import get_edge_features
from Featurizer.mol_features import get_molecular_features
from Featurizer.feature_maps import x_map

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

                # For printing 
                chirality_matrix = []
                hybridization_matrix = []
                bond_types_matrix = []

                for atom in mol.GetAtoms():
                    chirality = str(atom.GetChiralTag())
                    hybridization = str(atom.GetHybridization())
                    bond_types_connected = {str(b.GetBondType()) for b in atom.GetBonds()}

                    chirality_oh = [1 if chirality == c else 0 for c in x_map['chirality']]
                    hybrid_oh = [1 if hybridization == h else 0 for h in x_map['hybridization']]
                    bond_types_oh = [1 if b in bond_types_connected else 0 for b in x_map['bond_types_connected']]

                    chirality_matrix.append(chirality_oh)
                    hybridization_matrix.append(hybrid_oh)
                    bond_types_matrix.append(bond_types_oh)

                chirality_tensor = torch.tensor(chirality_matrix, dtype=torch.float)
                hybridization_tensor = torch.tensor(hybridization_matrix, dtype=torch.float)
                bond_types_tensor = torch.tensor(bond_types_matrix, dtype=torch.float)

                f.write("One-Hot Encoded Categorical Feature Matrices:\n")
                f.write("Chirality Matrix:\n")
                f.write(str(chirality_tensor) + "\n\n")

                f.write("Hybridization Matrix:\n")
                f.write(str(hybridization_tensor) + "\n\n")

                f.write("Bond Types Connected Matrix (Multi-Hot):\n")
                f.write(str(bond_types_tensor) + "\n\n")
                f.write("=" * 80 + "\n")

            except Exception as e:
                print(f"[ERROR] Failed on SMILES {smiles}: {e}")

    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()