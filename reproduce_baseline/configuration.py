import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, hamming_loss
from torch_scatter import scatter_softmax, scatter_sum
from typing import Tuple, Optional, List

import os
import pandas as pd

NUM_EPOCHS = 350
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
N_SPLITS = 2
SEED = 42
ALPHA = 0.1

def random_stratified_split(
    y: np.ndarray,
    sample_weights: np.ndarray,
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
    
    if w is None:
        w = np.ones_like(y)

    y_present = y != 0 & (w != 0)

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

def compute_alpha(loader, num_classes, device):
    total_counts = torch.zeros(num_classes, device=device)
    for _, labels in loader:
        labels = labels.to(device)
        total_counts += labels.sum(dim=0)
    # Avoid division by zero
    total_counts = torch.clamp(total_counts, min=1)
    pos_weight = (total_counts.sum() - total_counts) / total_counts
    return pos_weight

def focal_loss(logits, targets, gamma=2, alpha=None, reduction='mean', eps=1e-6):
    p = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)

    # if alpha is not None:
    #     alpha = alpha.to(logits.device)
    #     alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
    # else:
    #     alpha_factor = 1.0

    loss_pos = targets * (1 - p) ** gamma * torch.log(p)
    loss_neg = (1 - targets) * p ** gamma * torch.log(1 - p)
    loss = -alpha* (loss_pos + loss_neg)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def compute_class_counts(loader, num_classes, device):
    class_counts = torch.zeros(num_classes).to(device)
    for _, labels in loader:
        class_counts += labels.sum(dim=0)
    return class_counts.cpu().numpy()

def compute_per_class_metrics(y_true, y_pred, y_prob, label_names, output_dir=None):
    results = []
    for i, name in enumerate(label_names):
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=1)
        rec  = recall_score(y_true[:, i], y_pred[:, i], zero_division=1)
        f1   = f1_score(y_true[:, i], y_pred[:, i], zero_division=1)
        try:
            auc  = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            auc = float('nan')
        results.append((name, prec, rec, f1, auc))
    
    df = pd.DataFrame(results, columns=["Label", "Precision", "Recall", "F1", "AUROC"])
    print("\nPer-class metrics:")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)
        print(f"\nPer-class metrics saved to {output_dir}/per_class_metrics.csv")

    return df


def adaptive_focal_loss(logits, targets, gamma=1.5, alpha=0.6, alpha1=0.5, lambda_l2=1e-4,
                        reduction='mean', model=None, eps=1e-6):
    """
    Adaptive Focal Loss for multi-label classification with L2 regularization.
    
    Formulae:
        1. BCE Loss:
            L_bce = -[y * log(p) + (1 - y) * log(1 - p)]

        2. pt (sigmoid probability):
            pt = σ(x) for y = 1, and 1 - σ(x) for y = 0

        3. Focal Loss:
            L_focal = α * (1 - pt)^γ * L_bce

        4. Adaptive Loss:
            L = α1 * L_focal + (1 - α1) * L_bce

        5. Final Loss with L2 regularization:
            L_total = L + λ * L2
    """

    # Sigmoid probabilities
    probs = torch.sigmoid(logits).clamp(min=eps, max=1 - eps)

    # BCE loss (element-wise)
    bce_loss = - (targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))  # (12)

    # pt = probs when y=1, 1-probs when y=0  → used in Focal Loss scaling
    pt = targets * probs + (1 - targets) * (1 - probs)  # (13)

    # Focal scaling factor
    focal_weight = (1 - pt) ** gamma  # (14)

    # Apply focal scaling and alpha
    focal_loss = alpha * focal_weight * bce_loss  # (14)

    # Adaptive interpolation between focal loss and BCE
    loss = alpha1 * focal_loss + (1 - alpha1) * bce_loss  # (15)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    # Add L2 regularization if model is passed
    if model is not None and lambda_l2 > 0:
        l2_reg = sum(torch.norm(param, 2) ** 2 for param in model.parameters() if param.requires_grad)
        loss += lambda_l2 * l2_reg  # Final Loss

    return loss


def plot_conf_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def log_confusion_matrices(true, pred, labels):
    # Calculate accuracy per label
    label_accuracy = (true == pred).sum(axis=0) / true.shape[0]

    # Get indices for best and worst 20 labels by accuracy
    best = np.argsort(label_accuracy)[-20:]
    worst = np.argsort(label_accuracy)[:20]

    # Overall confusion matrix for all labels flattened
    plot_conf_matrix(confusion_matrix(true.flatten(), pred.flatten()), "All Labels Confusion Matrix", "total_cm.png")

    # Confusion matrix for best 20 labels flattened
    plot_conf_matrix(confusion_matrix(true[:, best].flatten(), pred[:, best].flatten()), "Best 20 Labels Confusion Matrix", "best_cm.png")

    # Confusion matrix for worst 20 labels flattened
    plot_conf_matrix(confusion_matrix(true[:, worst].flatten(), pred[:, worst].flatten()), "Worst 20 Labels Confusion Matrix", "worst_cm.png")

def visualize_label_distribution_per_fold(fold, label_array, set_name, label_names):
    label_counts = np.sum(label_array, axis=0)
    plt.figure(figsize=(12, 6))
    plt.bar(label_names, label_counts)
    plt.xticks(rotation=90)
    plt.ylabel("Count")
    plt.title(f"{set_name} Label Distribution - Split {split_num}")
    plt.tight_layout()
    os.makedirs("label_distributions", exist_ok=True)
    plt.savefig(f"label_distributions/{set_name.lower()}_split_{split_num}_distribution.png")
    plt.close()

def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data, nan_policy='omit')
    ci_range = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean, mean - ci_range, mean + ci_range

def plot_roc_curve(y_true, y_prob, label_names, title='ROC Curve', output_path=None):
    """
    Plots micro-averaged ROC curve and optionally per-class ROC curves.
    """
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'Micro-average ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    if output_path:
        plt.savefig(output_path)
    plt.show()

# class AttentionAggregation(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.attn_weights = nn.Linear(in_channels, 1)

#     def forward(self, x, batch):
#         # Compute attention scores
#         scores = self.attn_weights(x)  # (num_nodes, 1)
#         scores = scatter_softmax(scores, batch, dim=0)  # normalize over graphs

#         # Apply attention
#         x_weighted = x * scores  # element-wise multiply
#         global_feature = scatter_sum(x_weighted, batch, dim=0)  # sum over nodes per graph
#         return global_feature