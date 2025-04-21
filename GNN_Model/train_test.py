import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data, Batch
from sklearn.metrics import f1_score
import warnings
from rdkit import RDLogger
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter("ignore", category=UndefinedMetricWarning)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torchvision.ops import sigmoid_focal_loss

from GNN_Model.gcn_model import OdorClassifier
from Featurizer.from_smiles import from_smiles  # Feature extraction function

# Suppress RDKit warnings
rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.ERROR)

# Dataset class
class OdorDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles_list = smiles_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        data = from_smiles(smiles)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return data, label

# Molecule Feature Batching
class MoleculeDataBatch(Batch):
    @staticmethod
    def from_data_list(data_list):
        batch = Batch.from_data_list(data_list)
        
        # Handle molecular features separately
        mol_feats = torch.stack([d.mol_features for d in data_list])
        batch.mol_features = mol_feats
        
        return batch

# Custom collate function for PyTorch Geometric data
def collate_fn(batch):
    # Separate the graphs and labels
    graphs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    batched_graphs = MoleculeDataBatch.from_data_list(graphs)
    
    # Stack the labels
    batched_labels = torch.stack(labels)
    
    return batched_graphs, batched_labels

# Train Model
def train(model, train_loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # For multi-label classification, use sigmoid and threshold
        preds = torch.sigmoid(output) > 0.5
        all_preds.append(preds.cpu().numpy())
        all_labels.append(label.cpu().numpy())

    # Flatten lists of predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate accuracy score (multiclass classification)
    train_accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())

    return running_loss / len(train_loader), train_accuracy

def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            
            # For multi-label classification, use sigmoid and threshold
            preds = torch.sigmoid(output) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    # Flatten lists of predictions and labels
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate accuracy and F1 score
    val_accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)

    return val_accuracy, val_f1


def main():
    USE_FOCAL = False    # Toggle this to use foc

    # Load CSV data
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/OdorSmiles_Updated.csv', encoding='ISO-8859-1')

    # Separate out SMILES and CAS
    smiles_list = df['SMILES'].values
    labels_df = df.drop(columns=['SMILES', 'cas_number'])

    # Filter labels (odor descriptors) that appear in more than 10 molecules
    descriptor_counts = labels_df.sum(axis=0)
    valid_descriptors = descriptor_counts[descriptor_counts > 10].index
    filtered_labels = labels_df[valid_descriptors].values

    print(f"Original number of odors: {labels_df.shape[1]}")
    print(f"Remaining after thresholding: {len(valid_descriptors)}")
    print(f"Odors removed: {labels_df.shape[1] - len(valid_descriptors)}")

    # Calculate molecules with zero odor labels
    label_sums = filtered_labels.sum(axis=1)
    num_no_odor_molecules = np.sum(label_sums == 0)
    total_molecules = len(filtered_labels)
    
    # Cross-validation
    mskf = MultilabelStratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)


    for fold, (train_idx, val_idx) in enumerate(mskf.split(smiles_list, filtered_labels), 1):
        print(f"\n{'='*25} Fold {fold} {'='*25}")

        X_train, X_val = smiles_list[train_idx], smiles_list[val_idx]
        y_train, y_val = filtered_labels[train_idx], filtered_labels[val_idx]

        # Create datasets for each fold
        train_dataset = OdorDataset(X_train, y_train)
        val_dataset = OdorDataset(X_val, y_val)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = OdorClassifier(
            num_tasks=filtered_labels.shape[1],
            readout_dim=175,
            mlp_dims=[96, 63]
        )
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Loss function setup
        neg_counts = (filtered_labels == 0).sum(axis=0)
        pos_counts = (filtered_labels == 1).sum(axis=0)
        pos_weight = (neg_counts / (pos_counts + 1e-6)).astype(np.float32)
        pos_weight_tensor = torch.tensor(pos_weight).to(device)

        if USE_FOCAL:
            criterion = lambda output, target: sigmoid_focal_loss(output, target, alpha=0.5, gamma=1.0, reduction="mean")
        else:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # # Early Stopping and Checkpoint setup
        # best_f1 = 0
        # patience = 15
        # counter = 0
        # min_delta = 1e-4   
        # best_model_path = f"best_model_fold_{fold}.pt"

        # Training loop
        num_epochs = 100
        for epoch in range(1, num_epochs + 1):
            train_loss, train_accuracy = train(model, train_loader, device, optimizer, criterion)
            scheduler.step()

            # Print every 10 epochs and the first
            if epoch == 1 or epoch % 10 == 0:
                val_acc, val_f1 = evaluate(model, val_loader, device)
                print(
                    f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}\n"
                    f"|          Val   Acc : {val_acc:.4f} | F1 Score: {val_f1:.4f}"
                )

        #     # Early Stopping Check
        #     if val_f1 - best_f1 > min_delta:
        #         best_f1 = val_f1
        #         counter = 0
        #         torch.save(model.state_dict(), best_model_path)
        #         print(f"Saved best model at Epoch {epoch} with F1: {val_f1:.4f}")
        #     else:
        #         counter += 1
        #         print(f"No improvement for {counter} epoch(s)")
        #         if counter >= patience:
        #             print(f"Early stopping triggered after {patience} epochs.")
        #             break
        # # === Load Best Model and Final Evaluation ===
        # model.load_state_dict(torch.load(best_model_path))
        # final_val_acc, final_val_f1 = evaluate(model, val_loader, device)
        # print(f"\n Final Evaluation for Fold {fold}: Accuracy = {final_val_acc:.4f}, F1 Score = {final_val_f1:.4f}")

if __name__ == "__main__":
    main()
