import torch
import torch.nn.functional as F
import deepchem as dc
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from reproduce_baseline.configuration import focal_loss
from reproduce_baseline.AutoEncoder.UNetModel import OdorGCNUNet
from reproduce_baseline.Dataset import OdorDataset, collate_fn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruction_loss(node_out, x):
    # Example reconstruction loss: MSE between node features and predicted node output
    return F.mse_loss(node_out, x)

def train_one_epoch(model, optimizer, dataloader, alpha=0.1):
    """
    alpha: weight for reconstruction loss, (1-alpha) for graph loss
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data, labels in dataloader:
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        graph_out, node_out = model(data)
        labels = labels.float()
        x = data.x.float()

        # Graph-level focal loss
        graph_loss = focal_loss(graph_out, labels)
        # Node-level reconstruction loss
        recon_loss = reconstruction_loss(node_out, x)

        loss = (1 - alpha) * graph_loss + alpha * recon_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.sigmoid(graph_out).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels_np)

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    binarized_preds = (all_preds >= 0.35).astype(int)

    precision = precision_score(all_labels, binarized_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, binarized_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, binarized_preds, average='macro', zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        auroc = float('nan')

    return avg_loss, precision, recall, f1, auroc

def eval_one_epoch(model, dataloader, alpha=0.1):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            graph_out, node_out = model(data)
            labels = labels.float()
            x = data.x.float()

            graph_loss = focal_loss(graph_out, labels)
            recon_loss = reconstruction_loss(node_out, x)

            loss = (1 - alpha) * graph_loss + alpha * recon_loss
            total_loss += loss.item() * labels.size(0)

            preds = torch.sigmoid(graph_out).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels_np)

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    binarized_preds = (all_preds >= 0.35).astype(int)

    precision = precision_score(all_labels, binarized_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, binarized_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, binarized_preds, average='macro', zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        auroc = float('nan')

    return avg_loss, precision, recall, f1, auroc

def run_training_loop(batch_size=32, epochs=150, num_tasks=138, alpha=0.1):
    # Load the dataset
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv', encoding='ISO-8859-1')
    smiles_list = df["smiles"].values
    labels_df = df.drop(columns=["smiles", "descriptors"])
    label_names = labels_df.columns.tolist()
    labels = labels_df.values

    # Create the full dataset
    full_dataset = OdorDataset(smiles_list, labels)

    # Stratified K-Fold
    mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    fold_metrics = {
        'train_precision': [], 'train_recall': [], 'train_f1': [], 'train_auroc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auroc': []
    }

    for fold, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(full_dataset)), labels)):
        print(f"\n=== Fold {fold + 1} ===")

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = OdorGCNUNet(num_tasks=num_tasks, mlp_dims=[100, 70], alpha=0.5).to(device)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_f1 = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_p, train_r, train_f1, train_auroc = train_one_epoch(model, optimizer, train_loader, alpha=alpha)
            val_loss, val_p, val_r, val_f1, val_auroc = eval_one_epoch(model, val_loader, alpha=alpha)
            scheduler.step()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f"best_model_fold{fold + 1}.pt")

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}:")
                print(f"  Train Loss: {train_loss:.4f} | Precision: {train_p:.4f} | Recall: {train_r:.4f} | F1: {train_f1:.4f} | AUROC: {train_auroc:.4f}")
                print(f"  Val   Loss: {val_loss:.4f} | Precision: {val_p:.4f} | Recall: {val_r:.4f} | F1: {val_f1:.4f} | AUROC: {val_auroc:.4f}")

        fold_metrics['train_precision'].append(train_p)
        fold_metrics['train_recall'].append(train_r)
        fold_metrics['train_f1'].append(train_f1)
        fold_metrics['train_auroc'].append(train_auroc)

        fold_metrics['val_precision'].append(val_p)
        fold_metrics['val_recall'].append(val_r)
        fold_metrics['val_f1'].append(val_f1)
        fold_metrics['val_auroc'].append(val_auroc)

    print("\n=== Mean Metrics Across Folds ===")
    print(f"Train Precision: {np.mean(fold_metrics['train_precision']):.4f}")
    print(f"Train Recall:    {np.mean(fold_metrics['train_recall']):.4f}")
    print(f"Train F1:        {np.mean(fold_metrics['train_f1']):.4f}")
    print(f"Train AUROC:     {np.mean(fold_metrics['train_auroc']):.4f}")

    print(f"Val Precision:   {np.mean(fold_metrics['val_precision']):.4f}")
    print(f"Val Recall:      {np.mean(fold_metrics['val_recall']):.4f}")
    print(f"Val F1:          {np.mean(fold_metrics['val_f1']):.4f}")
    print(f"Val AUROC:       {np.mean(fold_metrics['val_auroc']):.4f}")


if __name__ == "__main__":
    run_training_loop(batch_size=32, epochs=150, num_tasks=138, alpha=0.1)

