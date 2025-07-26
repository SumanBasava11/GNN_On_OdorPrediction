import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, hamming_loss
from sklearn.exceptions import UndefinedMetricWarning
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, IterativeStratification, RepeatedMultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from rdkit import RDLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR 
import warnings
from sklearn.metrics import precision_recall_curve
from torch.utils.data import random_split, Subset

# Custom imports
from reproduce_baseline.configuration import *
from reproduce_baseline.model import OdorClassifier
# from different_models.RB_Best import OdorClassifier
from reproduce_baseline.Dataset import OdorDataset, collate_fn
from reproduce_baseline.box_plot import *
# from reproduce_baseline.MPNN.mpnn_model import OdorClassifier
# Suppress RDKit and sklearn warnings
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.simplefilter("ignore", category=UndefinedMetricWarning)
from typing import Tuple, Optional, List

def random_stratified_split(
    y: np.ndarray,
    frac_train: float = 0.8,
    frac_valid: float = 0.2,
    frac_test: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset indices into train/valid/test preserving multilabel stratification.

    Args:
        y (np.ndarray): binary multilabel matrix of shape (N, T)
        frac_train (float): fraction of samples to use for training
        frac_valid (float): fraction of samples to use for validation
        frac_test (float): fraction of samples to use for test
        seed (int, optional): random seed for reproducibility

    Returns:
        train_idx, valid_idx, test_idx: arrays of indices for each split
    """
    if seed is not None:
        np.random.seed(seed)
    
    # if w is None:
    #     w = np.ones_like(y)

    y_present = y != 0

    if y_present.ndim == 1:
        y_present = np.expand_dims(y_present, 1)
    elif y_present.ndim > 2:
        raise ValueError("y has more than 2 dimensions")

    n_tasks = y_present.shape[1]

    indices_for_task = [
        np.random.permutation(np.nonzero(y_present[:, i])[0])
        for i in range(n_tasks)
    ]
    count_for_task = np.array([len(x) for x in indices_for_task])
    train_target = np.round(frac_train * count_for_task).astype(int)
    valid_target = np.round(frac_valid * count_for_task).astype(int)
    test_target = np.round(frac_test * count_for_task).astype(int)

    train_counts = np.zeros(n_tasks, int)
    valid_counts = np.zeros(n_tasks, int)
    test_counts = np.zeros(n_tasks, int)

    set_target = [train_target, valid_target, test_target]
    set_counts = [train_counts, valid_counts, test_counts]
    set_inds: List[List[int]] = [[], [], []]
    assigned = set()

    max_count = np.max(count_for_task)

    for i in range(max_count):
        for task in range(n_tasks):
            indices = indices_for_task[task]
            if i < len(indices) and indices[i] not in assigned:
                index = indices[i]
                set_frac = [
                    1 if set_target[j][task] == 0 else set_counts[j][task] / set_target[j][task]
                    for j in range(3)
                ]
                set_index = np.argmin(set_frac)
                set_inds[set_index].append(index)
                assigned.add(index)
                set_counts[set_index] += y_present[index]

    # Fill remaining with negatives
    n_samples = y_present.shape[0]
    set_size = [
        int(np.round(n_samples * f)) for f in (frac_train, frac_valid, frac_test)
    ]

    s = 0
    for i in np.random.permutation(n_samples):
        if i not in assigned:
            while s < 2 and len(set_inds[s]) >= set_size[s]:
                s += 1
            set_inds[s].append(i)

    return (
        np.array(sorted(set_inds[0])),
        np.array(sorted(set_inds[1])),
        np.array(sorted(set_inds[2])),
    )

def train(model, loader, device, optimizer, scheduler, epoch, alpha):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output= model(data)
        loss = focal_loss(output, labels, gamma=2, alpha=0.25, reduction = 'sum')
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        probs = torch.sigmoid(output).detach().cpu().numpy()
        preds = (probs > 0.35).astype(int)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = np.vstack(all_preds)
    
    train_prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    train_rec = recall_score(y_true, y_pred, average='macro', zero_division=1)
    train_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    try:
        train_auroc = roc_auc_score(y_true, y_prob, average='macro')
    except ValueError:
        train_auroc = float('nan')

    print(f"Epoch {epoch:03d} | Train | Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1: {train_f1:.4f} | AUROC: {train_auroc:.4f}")

    return total_loss / len(loader), train_prec, train_rec, train_f1, train_auroc

def evaluate(model, loader, device, split = "Val"):
    model.eval()
    all_preds, all_labels, all_probs  = [], [], []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.cpu().numpy()
            logits = model(data)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs > 0.35).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
            
    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = np.vstack(all_preds)

    val_prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    val_rec = recall_score(y_true, y_pred, average='macro', zero_division=1)
    val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    try:
        val_roc_auc = roc_auc_score(y_true, y_prob, average='macro')
    except ValueError:
        val_roc_auc = float('nan')
    
    print(f"{split} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f} | AUROC: {val_roc_auc:.4f}")

    return val_prec, val_rec, val_f1, val_roc_auc, y_true, y_pred, y_prob

def main():
    df = pd.read_csv(
        'C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv',
        encoding='ISO-8859-1'
    )
    smiles = df["smiles"].values
    labels_df = df.drop(columns=["smiles", "descriptors"])
    labels = labels_df.values
    label_names = labels_df.columns.tolist()

    fold_y_trues = []
    fold_y_preds = []
    fold_y_probs = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_train_precisions, all_train_recalls, all_train_f1s, all_train_aurocs = [], [], [], []
    all_val_precisions, all_val_recalls, all_val_f1s, all_val_aurocs = [], [], [], []

    for fold in range(1, N_SPLITS+1):
        print(f"\nFold {fold}/{N_SPLITS} {'=' * 40}")
        full_dataset = OdorDataset(smiles, labels)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        train_idx, val_idx, _ = random_stratified_split(
            y= labels, frac_train=0.8, frac_valid=0.2, frac_test=0.0, seed=SEED + fold
        )

        print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")

        train_loader = DataLoader(
            OdorDataset(smiles[train_idx], labels[train_idx]),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            OdorDataset(smiles[val_idx], labels[val_idx]),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )

        alpha = compute_alpha(train_loader, num_classes=labels.shape[1], device=device)

        model = OdorClassifier(num_tasks=labels.shape[1], mlp_dims=[100, 70]).to(device)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=1e-3)  #1e-4
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

        best_val_f1 = 0
        best_val_prec, best_val_rec, best_val_auroc = 0, 0, 0

        best_train_prec, best_train_rec, best_train_f1, best_train_auroc = 0, 0, 0, 0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_prec, train_rec, train_f1, train_auroc = train(
                model, train_loader, device, optimizer, scheduler, epoch, alpha
            )

            val_prec, val_rec, val_f1, val_auroc, y_true, y_pred, y_prob = evaluate(
                model, val_loader, device, split=f"Validation Fold {fold}"
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_prec = val_prec
                best_val_rec = val_rec
                best_val_auroc = val_auroc

                # save the corresponding train metrics
                best_train_prec = train_prec
                best_train_rec = train_rec
                best_train_f1 = train_f1
                best_train_auroc = train_auroc

                # Save predictions of best fold
                fold_y_trues.append(y_true)
                fold_y_preds.append(y_pred)
                fold_y_probs.append(y_prob)

        all_val_precisions.append(best_val_prec)
        all_val_recalls.append(best_val_rec)
        all_val_f1s.append(best_val_f1)
        all_val_aurocs.append(best_val_auroc)

        all_train_precisions.append(best_train_prec)
        all_train_recalls.append(best_train_rec)
        all_train_f1s.append(best_train_f1)
        all_train_aurocs.append(best_train_auroc)

    # Print final stats across folds
    def print_stats(name, values):
        print(f"\n{name} stats across folds:")
        print(f"Mean:   {np.mean(values):.4f}")
        print(f"Std:    {np.std(values):.4f}")
        print(f"Min:    {np.min(values):.4f}")
        print(f"Median: {np.median(values):.4f}")
        print(f"Max:    {np.max(values):.4f}")

    def compute_confidence_interval(aurocs):
        mean = np.mean(aurocs)
        lower = np.percentile(aurocs, 2.5)
        upper = np.percentile(aurocs, 97.5)
        return mean, lower, upper

    print_stats("Train Precision", all_train_precisions)
    print_stats("Train Recall", all_train_recalls)
    print_stats("Train F1", all_train_f1s)
    mean_auc, ci_lower, ci_upper = compute_confidence_interval(all_train_aurocs)
    print(f"\nTrain AUROC:\nMean: {mean_auc:.4f}, 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

    print_stats("Validation Precision", all_val_precisions)
    print_stats("Validation Recall", all_val_recalls)
    print_stats("Validation F1", all_val_f1s)
    mean_auc, ci_lower, ci_upper = compute_confidence_interval(all_val_aurocs)
    print(f"\nValidation AUROC:\nMean: {mean_auc:.4f}, 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

    # Combine predictions across folds
    y_trues = np.concatenate(fold_y_trues, axis=0)
    y_preds = np.concatenate(fold_y_preds, axis=0)
    y_probs = np.concatenate(fold_y_probs, axis=0)

    # Compute per-label metrics
    per_label_f1 = []
    per_label_auroc = []
    descriptor_metrics = []

    for i in range(y_trues.shape[1]):
        try:
            f1 = f1_score(y_trues[:, i], y_preds[:, i], zero_division=1)
            auroc = roc_auc_score(y_trues[:, i], y_probs[:, i])
            support = int(np.sum(y_trues[:, i]))
        except ValueError:
            f1, auroc = float('nan'), float('nan'), 0
        # per_label_f1.append(f1)
        # per_label_auroc.append(auroc)
        descriptor_metrics.append((label_names[i], f1, auroc, support))

    # Sort by F1 (desc), then AUROC (desc)
    descriptor_metrics_sorted = sorted(
    descriptor_metrics, key=lambda x: (x[2], x[1]), reverse=True
    )

    # Save results to a single .txt file
    output_path = "C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/reproduce_baseline/per_descriptors_metrics.txt"
    with open(output_path, "w") as f:
        f.write("Descriptor\tF1\tAUROC\tSupport\n")
        for name, f1, auc, support in descriptor_metrics_sorted:
            f.write(f"{name}\t{f1:.4f}\t{auc:.4f}\t{support}\n")

    print(f"\nSaved per-descriptor metrics to: {output_path}")
if __name__ == "__main__":
    main()