import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from rdkit import RDLogger
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from reproduce_baseline.configuration import *
from reproduce_baseline.model import OdorClassifier
from reproduce_baseline.Dataset import OdorDataset, collate_fn
from train_utils.label_distribution import visualize_label_distribution_per_fold
from reproduce_baseline.metrics_report import write_metrics_report

# Suppress RDKit + sklearn warnings
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.simplefilter("ignore", category=UndefinedMetricWarning)

NUM_REPEATS = 5  # Number of repeated random splits

def train(model, loader, device, optimizer, criterion, epoch, label_names):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for data, labels in loader:
        if data.num_graphs == 1:
            continue
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        # loss = focal_loss(output, labels, gamma=1.5)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = (torch.sigmoid(output) > 0.4).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    train_acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    train_prec = precision_score(y_true, y_pred, average='micro', zero_division=1)
    train_rec = recall_score(y_true, y_pred, average='micro', zero_division=1)
    train_f1 = f1_score(y_true, y_pred, average='micro', zero_division=1)

    try:
        train_auroc = roc_auc_score(y_true, y_pred, average='micro')
    except ValueError:
        train_auroc = float('nan') 

    print(f"Epoch {epoch:03d} | Train | Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1: {train_f1:.4f} | AUROC: {train_auroc:.4f}")
    
    log_confusion_matrices(y_true, y_pred, label_names)
    
    return total_loss / len(loader), train_acc, train_prec, train_rec, train_f1, train_auroc

def evaluate(model, loader, device, label_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = labels.cpu().numpy()
            preds = (probs > 0.4).astype(int)
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)

    val_acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    val_prec = precision_score(y_true, y_pred, average='micro', zero_division=1)
    val_rec = recall_score(y_true, y_pred, average='micro', zero_division=1)
    val_f1 = f1_score(y_true, y_pred, average='micro', zero_division=1)

    try:
        val_roc_auc = roc_auc_score(y_true, y_prob, average='micro')
    except ValueError:
        val_roc_auc = float('nan')

    log_confusion_matrices(y_true, y_pred, label_names)

    return val_acc, val_prec, val_rec, val_f1, val_roc_auc, y_true, y_pred, y_prob

def print_stats(name, values):
    print(f"\n{name} stats across {NUM_REPEATS} splits:")
    print(f"Mean:   {np.mean(values):.4f}")
    print(f"Std:    {np.std(values):.4f}")
    print(f"Min:    {np.min(values):.4f}")
    print(f"Median: {np.median(values):.4f}")
    print(f"Max:    {np.max(values):.4f}")

def main():
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/raw_Balanced_OdorSmiles_Top138.csv',encoding='ISO-8859-1')

    smiles = df["SMILES"].values
    labels_df = df.drop(columns=["SMILES", "cas_number"])
    label_names = labels_df.columns.tolist()
    labels = labels_df.values

    pos_counts = (labels == 1).sum(axis=0)
    neg_counts = (labels == 0).sum(axis=0)
    pos_weights = torch.tensor((neg_counts / (pos_counts + 1e-6)), dtype=torch.float32)

    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_aurocs = []

    # For training
    all_train_precisions = []
    all_train_recalls = []
    all_train_f1s = []
    all_train_aurocs = []

    for split_num in range(1, NUM_REPEATS + 1):
        print(f"\nSplit {split_num}/{NUM_REPEATS} {'=' * 40}")

        train_smiles, val_smiles, train_labels, val_labels = train_test_split(
            smiles, labels, test_size=0.2, random_state=SEED + split_num
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OdorClassifier(num_tasks=labels.shape[1], mlp_dims=[96, 63]).to(device)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

        train_loader = DataLoader(
            OdorDataset(train_smiles, train_labels),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            OdorDataset(val_smiles, val_labels),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc, train_prec, train_rec, train_f1, train_auroc = train(model, train_loader, device, optimizer, criterion, epoch, label_names)

        # Save last epoch's train metrics
        all_train_precisions.append(train_prec)
        all_train_recalls.append(train_rec)
        all_train_f1s.append(train_f1)
        all_train_aurocs.append(train_auroc)

        val_acc, val_prec, val_rec, val_f1, val_roc_auc, y_true, y_pred, y_prob = evaluate(model, val_loader, device, label_names)
        print(f"Validation Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1_micro: {val_f1:.4f} | AUROC: {val_roc_auc:.4f}")

        # Write the per-label metrics report file for this fold
        write_metrics_report(y_true, y_prob, y_pred, label_names)
        
        all_precisions.append(val_prec)
        all_recalls.append(val_rec)
        all_f1s.append(val_f1)
        all_aurocs.append(val_roc_auc)

        # Print running stats after this fold
        print(f"\nStats after Split {split_num}:")
        print(f"Precision - Mean: {np.mean(all_precisions):.4f}, Std: {np.std(all_precisions):.4f}")
        print(f"Recall    - Mean: {np.mean(all_recalls):.4f}, Std: {np.std(all_recalls):.4f}")
        print(f"F1_micro  - Mean: {np.mean(all_f1s):.4f}, Std: {np.std(all_f1s):.4f}")
        print(f"AUROC     - Mean: {np.mean(all_aurocs):.4f}, Std: {np.std(all_aurocs):.4f}")

    def print_stats(name, values):
        print(f"\n{name} stats across {N_SPLITS} Splits:")
        print(f"Mean:   {np.mean(values):.4f}")
        print(f"Std:    {np.std(values):.4f}")
        print(f"Min:    {np.min(values):.4f}")
        print(f"Median: {np.median(values):.4f}")
        print(f"Max:    {np.max(values):.4f}")

    # Print Train Stats
    print_stats("Train Precision", all_train_precisions)
    print_stats("Train Recall", all_train_recalls)
    print_stats("Train F1_micro", all_train_f1s)
    mean_train_auc, ci_lower, ci_upper = compute_confidence_interval(all_train_aurocs)
    print(f"\nTrain AUROC (micro) across {N_SPLITS} Splits:")
    print(f"Mean AUROC: {mean_train_auc:.4f}")
    print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")


    # Print Validation Stats
    print_stats("Validation Precision", all_precisions)
    print_stats("Validation Recall", all_recalls)
    print_stats("Validation F1_micro", all_f1s)
    mean_auc, lower_ci, upper_ci = compute_confidence_interval(all_aurocs)
    print(f"\nValidation AUROC (micro) across {N_SPLITS} Splits:")
    print(f"Mean AUROC: {mean_auc:.4f}")
    print(f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})")


if __name__ == "__main__":
    main()
