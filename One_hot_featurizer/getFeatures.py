from rdkit import Chem
import torch
from torch_geometric.data import Data
import pandas as pd
from One_hot_featurizer.feature_maps import x_map, e_map, mol_map  # assumes you saved your maps in one_hot_maps.py
from rdkit.Chem import Descriptors
import torch.nn.functional as F

def one_hot(value, choices):
    if isinstance(choices[0], bool):
        idx = int(bool(value))
    else:
        # Try casting to str if value looks like a RDKit enum string
        if isinstance(value, str) and value in choices:
            idx = choices.index(value)
        elif isinstance(value, int) and value in choices:
            idx = choices.index(value)
        else:
            # fallback if value is invalid
            idx = 0
    return F.one_hot(torch.tensor(idx), num_classes=len(choices)).float()

def one_hot_dict(value, mapping):
    """For dict-based maps (like chirality or hybridization)."""
    vec = torch.zeros(len(mapping), dtype=torch.float32)
    if value in mapping:
        idx = mapping[value]
        vec[idx] = 1.0
    return vec

def get_node_features_onehot(atom: Chem.Atom) -> torch.Tensor:
    feats = []
    feats.append(one_hot(atom.GetAtomicNum(), x_map['atomic_num']))
    feats.append(one_hot(atom.GetDegree(), x_map['degree']))
    feats.append(one_hot(atom.GetFormalCharge(), x_map['formal_charge']))
    feats.append(one_hot(atom.GetTotalNumHs(), x_map['num_hs']))
    feats.append(one_hot(atom.GetNumRadicalElectrons(), x_map['num_radical_electrons']))
    feats.append(one_hot(atom.GetTotalValence(), x_map['valence']))
    feats.append(one_hot_dict(atom.GetIsAromatic(), x_map['is_aromatic']))
    feats.append(one_hot_dict(atom.IsInRing(), x_map['is_in_ring']))
    ring_sizes = [len(r) for r in atom.GetOwningMol().GetRingInfo().AtomRings() if atom.GetIdx() in r]
    smallest_ring = min(ring_sizes) if ring_sizes else 0
    feats.append(one_hot(smallest_ring, x_map['smallest_ring']))
    feats.append(one_hot_dict(str(atom.GetChiralTag()), x_map['chirality']))
    feats.append(one_hot_dict(str(atom.GetHybridization()), x_map['hybridization']))
    feats.append(one_hot(str(atom.GetBonds()[0].GetBondType()) if atom.GetBonds() else 'SINGLE', x_map['bond_types_connected']))
    return torch.cat(feats)

def get_edge_features_onehot(bond: Chem.Bond) -> torch.Tensor:
    feats = []
    feats.append(one_hot(str(bond.GetStereo()), e_map['stereo']))
    feats.append(one_hot_dict(bond.GetIsConjugated(), e_map['is_conjugated']))
    return torch.cat(feats)

def get_molecular_features_onehot(mol: Chem.Mol) -> torch.Tensor:
    """
    Compute molecular descriptors from RDKit and return a feature vector.
    """
    mol_props = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_rings': mol.GetRingInfo().NumRings(),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_H_bond_donors': Descriptors.NumHDonors(mol),
        'num_H_bond_acceptors': Descriptors.NumHAcceptors(mol),
        'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
        'formal_charge': Chem.GetFormalCharge(mol),
        'complexity': Descriptors.FractionCSP3(mol),  # or another measure
    }

    feats = []
    for key, choices in mol_map.items():
        value = mol_props.get(key, 0)
        if choices is None:
            feats.append(torch.tensor([float(value)], dtype=torch.float32))
        else:
            try:
                feats.append(one_hot(int(value), choices))
            except ValueError:
                feats.append(torch.zeros(len(choices), dtype=torch.float32))
    return torch.cat(feats)

def from_smiles_onehot(smiles: str, with_hydrogen: bool = False, kekulize: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[Invalid SMILES] {smiles}")
        return None

    try:
        if with_hydrogen:
            mol = Chem.AddHs(mol)
        if kekulize:
            Chem.Kekulize(mol)

        # Nodes
        node_feats = []
        for atom in mol.GetAtoms():
            node_feats.append(get_node_features_onehot(atom))
        x = torch.stack(node_feats)

        # Edges
        edge_index = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
            feat = get_edge_features_onehot(bond)
            edge_attrs.append(feat)
            edge_attrs.append(feat)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs)

        # Molecular features
        mol_feat = get_molecular_features_onehot(mol)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles,
            mol_features=mol_feat
        )

        return data

    except Exception as e:
        print(f"[from_smiles_onehot ERROR] {smiles} => {e}")
        return None

# Main function to features and print them to a file
def main():
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv', encoding='ISO-8859-1')
    output_path = "Featurizer/smiles_features_output.txt"

    with open(output_path, "w") as f:
        for index, row in df.iterrows():
            smiles = row['smiles']

            data = from_smiles_onehot(smiles)
            if data is None:
                continue

            try:
                # Get RDKit mol object from smiles for edge features
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                f.write(f"\nFeatures for SMILES: {smiles}\n")
                f.write("-" * 60 + "\n")
                f.write("Node Feature Matrix:\n")
                f.write(str(data.x) + "\n\n")

                f.write("Edge Feature Matrix:\n")
                f.write(str(data.edge_attr) + "\n\n")

                f.write("Molecular Features:\n")
                f.write(str(data.mol_features) + "\n\n")

                f.write("=" * 80 + "\n")

            except Exception as e:
                print(f"[ERROR] Failed on SMILES {smiles}: {e}")
        
    print(f"Output saved to: {output_path}")
    

if __name__ == "__main__":
    main()