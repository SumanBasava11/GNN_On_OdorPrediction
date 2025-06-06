import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_SPLITS = 5
SEED = 42

def focal_loss(logits, targets, gamma=1.5, alpha=None, reduction='mean', eps=1e-6):
    p = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)

    if alpha is not None:
        alpha = alpha.to(logits.device)
        alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
    else:
        alpha_factor = 1.0

    loss_pos = targets * (1 - p) ** gamma * torch.log(p)
    loss_neg = (1 - targets) * p ** gamma * torch.log(1 - p)
    loss = -alpha_factor * (loss_pos + loss_neg)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def adaptive_focal_loss(logits, targets, gamma=2.0, alpha=0.6, alpha1=0.5, lambda_l2=1e-4,
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
