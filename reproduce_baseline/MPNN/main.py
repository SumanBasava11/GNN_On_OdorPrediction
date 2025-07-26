import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, hamming_loss
from sklearn.exceptions import UndefinedMetricWarning
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from rdkit import RDLogger
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR

from reproduce_baseline.MPNN.mpnnModel import MPNNPOM_PyG
from reproduce_baseline.Dataset import OdorDataset, collate_fn
from reproduce_baseline.configuration import *

# Suppress warnings
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.simplefilter("ignore", category=UndefinedMetricWarning)

# def compute_alpha(train_loader, num_classes, device):
#     label_sum = torch.zeros(num_classes).to(device)
#     for _, labels in train_loader:
#         label_sum += labels.sum(dim=0)
#     alpha = 1.0 / (label_sum + 1e-6)
#     alpha = alpha / alpha.sum()
#     return alpha

def train(model, loader, device, optimizer, epoch):
    model.train()
    total_loss = 0
    all_probs, all_labels, all_preds = [], [], []

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        proba, logits, embeddings = model(data)
        logits = logits.squeeze(-1)
        optimizer.zero_grad()
        alpha = torch.tensor(0.5).to(logits.device)
        loss = focal_loss(logits, labels, alpha=alpha, gamma=2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu()
        preds = (probs >= 0.35).float()
        all_probs.append(probs.numpy())
        all_preds.append(preds.numpy())
        all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = np.vstack(all_preds)

    train_acc = 1 - hamming_loss(y_true, y_pred)
    train_prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    train_rec = recall_score(y_true, y_pred, average='macro', zero_division=1)
    train_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    try:
        train_auroc = roc_auc_score(y_true, y_prob, average='macro')
    except ValueError:
        train_auroc = float('nan')

    print(f"Epoch {epoch:03d} | Train | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f} | AUROC: {train_auroc:.4f}")
    return total_loss / len(loader), train_acc, train_prec, train_rec, train_f1, train_auroc

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            proba, logits, embeddings = model(data)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= 0.35).float()

            all_probs.append(probs.numpy())
            all_preds.append(preds.numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = np.vstack(all_preds)

    val_acc = 1 - hamming_loss(y_true, y_pred)
    val_prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    val_rec = recall_score(y_true, y_pred, average='macro', zero_division=1)
    val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    try:
        val_auroc = roc_auc_score(y_true, y_prob, average='macro')
    except ValueError:
        val_auroc = float('nan')

    return val_acc, val_prec, val_rec, val_f1, val_auroc, y_true, y_pred, y_prob

def main():
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv', encoding='ISO-8859-1')
    smiles = df["smiles"].values
    labels = df.drop(columns=["smiles", "descriptors"]).values
    n_tasks = labels.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_train_metrics, all_val_metrics = [], []

    for fold, (train_idx, val_idx) in enumerate(mskf.split(smiles, labels), 1):
        print(f"\nFold {fold}/{N_SPLITS} {'=' * 40}")

        train_loader = DataLoader(OdorDataset(smiles[train_idx], labels[train_idx]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(OdorDataset(smiles[val_idx], labels[val_idx]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # alpha = compute_alpha(train_loader, num_classes=n_tasks, device=device)

        model = MPNNPOM_PyG(n_tasks=n_tasks).to(device)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = train(model, train_loader, device, optimizer, epoch )
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)

        print(f"Fold {fold} | Val Acc: {val_metrics[0]:.4f} | Prec: {val_metrics[1]:.4f} | Rec: {val_metrics[2]:.4f} | F1: {val_metrics[3]:.4f} | AUROC: {val_metrics[4]:.4f}")

    # Aggregate and report cross-validation results
    def summarize_metrics(metrics, name):
        metrics = np.array(metrics)
        print(f"\n{name} Metrics across {N_SPLITS} folds:")
        means = metrics.mean(axis=0)
        for i, metric_name in enumerate(['Loss', 'Acc', 'Prec', 'Rec', 'F1', 'AUROC']):
            print(f"{metric_name}: {means[i]:.4f}")

    summarize_metrics(all_train_metrics, "Train")
    summarize_metrics(all_val_metrics, "Validation")

if __name__ == "__main__":
    main()
