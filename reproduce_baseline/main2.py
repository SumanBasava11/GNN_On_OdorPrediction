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
import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
from reproduce_baseline.configuration import *
from reproduce_baseline.model import OdorClassifier
from reproduce_baseline.Dataset import OdorDataset, collate_fn
# from reproduce_baseline.metrics_report import write_metrics_report
# from reproduce_baseline.metrics_utils import focal_loss, log_confusion_matrices, compute_confidence_interval
from reproduce_baseline.box_plot import *

# Suppress RDKit and sklearn warnings
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.simplefilter("ignore", category=UndefinedMetricWarning)

# Metrics for boxplot
all_f1s_per_label = []  # List of shape (num_folds, num_labels)
all_aurocs_per_label = []

def train(model, loader, device, optimizer, epoch, label_names):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for data, labels in loader:
        if data.num_graphs == 1:
            continue
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = focal_loss(output, labels, gamma=2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        probs = torch.sigmoid(output).detach().cpu().numpy()
        preds = (probs > 0.4).astype(int)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_prob = np.vstack(all_probs)

    train_acc = 1 - hamming_loss(y_true, y_pred)
    train_prec = precision_score(y_true, y_pred, average='micro', zero_division=1)
    train_rec = recall_score(y_true, y_pred, average='micro', zero_division=1)
    train_f1 = f1_score(y_true, y_pred, average='micro', zero_division=1)
    try:
        train_auroc = roc_auc_score(y_true, y_prob, average='micro')
    except ValueError:
        train_auroc = float('nan')

    print(f"Epoch {epoch:03d} | Train | Acc: {train_acc:.4f} | Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1: {train_f1:.4f} | AUROC: {train_auroc:.4f}")
    return total_loss / len(loader), train_acc, train_prec, train_rec, train_f1, train_auroc

def evaluate(model, loader, device, label_names):
    model.eval()
    all_preds, all_labels, all_probs  = [], [], []

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

    val_acc = 1 - hamming_loss(y_true, y_pred)
    val_prec = precision_score(y_true, y_pred, average='micro', zero_division=1)
    val_rec = recall_score(y_true, y_pred, average='micro', zero_division=1)
    val_f1 = f1_score(y_true, y_pred, average='micro', zero_division=1)
    try:
        val_roc_auc = roc_auc_score(y_true, y_prob, average='micro')
    except ValueError:
        val_roc_auc = float('nan')

    # Per-label F1 and AUROC for boxplot
    label_f1s = f1_score(y_true, y_pred, average=None, zero_division=1)
    try:
        label_aurocs = roc_auc_score(y_true, y_prob, average=None)
    except ValueError:
        label_aurocs = [float('nan')] * len(label_names)

    all_f1s_per_label.append(label_f1s)
    all_aurocs_per_label.append(label_aurocs)

    return val_acc, val_prec, val_rec, val_f1, val_roc_auc, y_true, y_pred, y_prob

def main():
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(Saturated)SoS_Top138.csv', encoding='ISO-8859-1')
    smiles = df["SMILES"].values
    labels_df = df.drop(columns=["SMILES", "cas_number"])
    label_names = labels_df.columns.tolist()
    labels = labels_df.values

    pos_counts = (labels == 1).sum(axis=0)
    neg_counts = (labels == 0).sum(axis=0)
    pos_weights = torch.tensor((neg_counts / (pos_counts + 1e-6)), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_precisions, all_recalls, all_f1s, all_aurocs, all_val_accuracies = [], [], [], [], []
    all_train_precisions, all_train_recalls, all_train_f1s, all_train_aurocs, all_train_accuracies = [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(mskf.split(smiles, labels), 1):
        print(f"\nFold {fold}/{N_SPLITS} {'=' * 40}")

        model = OdorClassifier(num_tasks=labels.shape[1], mlp_dims=[96, 63]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=1e-3)

        train_loader = DataLoader(
            OdorDataset(smiles[train_idx], labels[train_idx]),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            OdorDataset(smiles[val_idx], labels[val_idx]),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc, train_prec, train_rec, train_f1, train_auroc = train(
                model, train_loader, device, optimizer, epoch, label_names
            )

        all_train_accuracies.append(train_acc)
        all_train_precisions.append(train_prec)
        all_train_recalls.append(train_rec)
        all_train_f1s.append(train_f1)
        all_train_aurocs.append(train_auroc)

        val_acc, val_prec, val_rec, val_f1, val_roc_auc, y_true, y_pred, y_prob = evaluate(
            model, val_loader, device, label_names
        )
        all_val_accuracies.append(val_acc)
        all_precisions.append(val_prec)
        all_recalls.append(val_rec)
        all_f1s.append(val_f1)
        all_aurocs.append(val_roc_auc)

        print(f"Fold {fold} | Val Acc: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f} | AUROC: {val_roc_auc:.4f}")
        # write_metrics_report(y_true, y_prob, y_pred, label_names, filename=f"metrics_fold_{fold}.txt")

    def print_stats(name, values):
        print(f"\n{name} stats across {N_SPLITS} folds:")
        print(f"Mean:   {np.mean(values):.4f}")
        print(f"Std:    {np.std(values):.4f}")
        print(f"Min:    {np.min(values):.4f}")
        print(f"Median: {np.median(values):.4f}")
        print(f"Max:    {np.max(values):.4f}")

    print_stats("Train Accuracy", all_train_accuracies)
    print_stats("Train Precision", all_train_precisions)
    print_stats("Train Recall", all_train_recalls)
    print_stats("Train F1", all_train_f1s)
    mean_auc, ci_lower, ci_upper = compute_confidence_interval(all_train_aurocs)
    print(f"\nTrain AUROC:\nMean: {mean_auc:.4f}, 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

    print_stats("Validation Accuracy", all_val_accuracies)
    print_stats("Validation Precision", all_precisions)
    print_stats("Validation Recall", all_recalls)
    print_stats("Validation F1", all_f1s)
    mean_auc, ci_lower, ci_upper = compute_confidence_interval(all_aurocs)
    print(f"\nValidation AUROC:\nMean: {mean_auc:.4f}, 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

    # Convert per-label metric lists to arrays
    f1s_per_label_array = np.array(all_f1s_per_label)
    aurocs_per_label_array = np.array(all_aurocs_per_label)

    # Plot per-label validation boxplots
    plot_boxplots_per_label(label_names, f1s_per_label_array, aurocs_per_label_array, k=30)

if __name__ == "__main__":
    main()
