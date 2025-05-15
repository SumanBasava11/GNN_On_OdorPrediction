from rdkit import Chem
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from Featurizer.node_features import get_node_features
from Featurizer.edge_features import get_edge_features
from Featurizer.mol_features import get_molecular_features
# from Featurizer.normalization_utils import MolFeatureNormalizer, NodeFeatureNormalizer

df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/Balanced_OdorSmiles_Top30.csv', encoding='ISO-8859-1')

# # Fit the normalizers
# mol_feats = []
# node_feats = []

# # Initialize normalizers
# mol_norm = MolFeatureNormalizer()
# node_norm = NodeFeatureNormalizer(continuous_indices=[0, 1, 2, 3, 4, 5, 8])  # Define indices of features to normalize

# # Function to fit normalizers on the dataset
# def fit_normalizers(smiles_list):
#     mol_feats = []
#     node_feats = []

#     # Collect all molecular and node features from the dataset
#     for smi in smiles_list:
#         mol = Chem.MolFromSmiles(smi)
#         if mol is None:
#             continue
#         mol_feats.append(get_molecular_features(mol))
#         node_feats.append(get_node_features(mol))

#     # Fit the normalizers
#     mol_norm.fit(mol_feats)
#     node_norm.fit(node_feats)

# # Fit normalizers on the full dataset of SMILES
# smiles_list = df['SMILES'].values
# fit_normalizers(smiles_list)

def from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[Invalid SMILES] {smiles}")
        return None
    try:
        if with_hydrogen:
            mol = Chem.AddHs(mol)
        if kekulize:
            Chem.Kekulize(mol)

        node_feat = get_node_features(mol)
        # node_feat = node_norm.transform(node_feat)  # Normalize node features

        edge_index, edge_attr = get_edge_features(mol, num_nodes=node_feat.size(0))

        mol_feat = get_molecular_features(mol)
        # mol_feat = mol_norm.transform(mol_feat)  # Normalize molecular features

        data = Data(
            x=node_feat, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            smiles=smiles
        )
        
        data.mol_features = mol_feat

        return data
        
    except Exception as e:
        print(f"[from_smiles ERROR] {smiles} => {e}")
        return None
