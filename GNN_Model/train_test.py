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
    
    # Batch the graphs
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
    # Load CSV data
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/OdorSmiles_Updated.csv', encoding='ISO-8859-1')

    # Separate out SMILES and CAS
    smiles_list = df['SMILES'].values
    labels_df = df.drop(columns=['SMILES', 'cas_number'])

    # Filter labels (odor descriptors) that appear in more than 10 molecules
    descriptor_counts = labels_df.sum(axis=0)
    valid_descriptors = descriptor_counts[descriptor_counts > 10].index
    filtered_labels = labels_df[valid_descriptors].values

    # Filter SMILES accordingly
    smiles_list = df['SMILES'].values

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(smiles_list, filtered_labels, test_size=0.1, random_state=42)

    # Dataset and DataLoader
    train_dataset = OdorDataset(X_train, y_train)
    val_dataset = OdorDataset(X_val, y_val)

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
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, device, optimizer, criterion)
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        # Validate after every epoch
        if (epoch + 1) % 10 == 0:
            val_accuracy, val_f1 = evaluate(model, val_loader, device)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}')

if __name__ == "__main__":
    main()
