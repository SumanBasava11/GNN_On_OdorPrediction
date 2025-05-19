import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torchvision.ops import sigmoid_focal_loss
import warnings
from torchinfo import summary
from rdkit import RDLogger
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter("ignore", category=UndefinedMetricWarning)
from train_utils.metrics import save_per_label_metrics
from train_utils.utils import save_label_distribution
from train_utils.config import *
from GNN_Model.gcn_model import OdorClassifier
from train_utils.dataset import OdorDataset, collate_fn
from train_utils.train_eval import train, evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from train_utils.label_distribution import visualize_label_distribution_per_fold
from torchinfo import summary
from sklearn.model_selection import KFold

# Suppress RDKit warnings
rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.ERROR)


def focal_loss(logits, targets, gamma=1.5, reduction='mean', eps=1e-6):
    """
    Args:
        logits: Raw model outputs (before sigmoid), shape [batch_size, num_classes]
        targets: Binary labels (0 or 1), same shape
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
    """
    # Apply sigmoid to get predicted probabilities
    p = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps) 
    
    # Compute the cross-entropy components for each label
    loss_pos = targets * (1 - p) ** gamma * torch.log(p) 
    loss_neg = (1 - targets) * p ** gamma * torch.log(1 - p)
    
    # Total loss (no alpha)
    loss = -(loss_pos + loss_neg)

    # Apply reduction method
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def compute_auc_per_label(model, val_loader, device, label_names, output_path="auc-roc/auc_roc.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 

    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for graphs, labels in val_loader:
            graphs = graphs.to(device)
            labels = labels.to(device)

            outputs = model(graphs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

def main():
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/Balanced_OdorSmiles_Top100.csv', encoding='ISO-8859-1')
    
    # SMILES and labels
    smiles = df["SMILES"].values
    labels_df = df.drop(columns=["SMILES", "cas_number"])
    label_names = labels_df.columns.tolist()
    labels = labels_df.values

    pos_counts = (labels == 1).sum(axis=0)
    neg_counts = (labels == 0).sum(axis=0)
    pos_weights = torch.tensor((neg_counts / (pos_counts + 1e-6)), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    # kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Lists to store metrics across folds
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for fold, (train_idx, val_idx) in enumerate(mskf.split(smiles, labels), 1):
    # for fold, (train_idx, val_idx) in enumerate(kf.split(smiles), 1): 
        print(f"\nFold {fold}/{N_SPLITS} {'=' * 40}")

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        # Visualize per-fold label distribution
        visualize_label_distribution_per_fold(fold, train_labels, "Train", label_names)
        visualize_label_distribution_per_fold(fold, val_labels, "Validation", label_names)

        # Model initialization
        model = OdorClassifier(num_tasks=labels.shape[1], mlp_dims=[96, 63]).to(device)
        print(model)

        # parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

        train_loader = DataLoader(OdorDataset(smiles[train_idx], labels[train_idx]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(OdorDataset(smiles[val_idx], labels[val_idx]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # Choose loss function
        if USE_FOCAL:
            criterion = lambda out, tgt: sigmoid_focal_loss(out, tgt, alpha=0.5, gamma=2, reduction="mean")
        else:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

        # print(f"Train set size: {len(train_loader.dataset)}")
        # print(f"Batch size: {BATCH_SIZE}")
        # print(f"Number of batches in train_loader: {len(train_loader)}")
        # print(f"Validation set size: {len(val_loader.dataset)}")
        # print(f"Number of batches in val_loader: {len(val_loader)}")
        
        # --- Collect batch-wise label distribution for first 50 batches ---
        # batch_label_counts = []
        # for i, (data, labels_batch) in enumerate(train_loader):
        #     if i >= 50:  # Limit to first 50 batches
        #         break
        #     batch_counts = labels_batch.sum(dim=0).numpy()
        #     batch_label_counts.append(batch_counts)

        # # Convert to DataFrame
        # batch_df = pd.DataFrame(batch_label_counts, columns=label_names)
        # batch_df["Batch"] = batch_df.index + 1

        # # Melt for histogram
        # melted = batch_df.melt(id_vars=["Batch"], var_name="Label", value_name="Positive Samples")

        # # Plot histogram per label
        # plt.figure(figsize=(14, 6))
        # sns.histplot(data=melted, x="Positive Samples", hue="Label", element="step", stat="count", common_norm=False, bins=range(0, BATCH_SIZE + 1, 1), multiple="stack")
        # plt.title(f"Histogram of Positive Samples per Label across Batches (Fold {fold})")
        # plt.xlabel("Number of Positive Samples in a Batch")
        # plt.ylabel("Frequency")
        # plt.tight_layout()
        # os.makedirs("batch_histograms", exist_ok=True)
        # plt.savefig(f"batch_histograms/batchwise_label_distribution_fold{fold}.png")
        # plt.close()
        # print(f"Histogram saved to: batch_histograms/batchwise_label_distribution_fold{fold}.png")
        
        # Start training loop
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc, train_prec, train_rec, train_f1_macro, train_f1_micro = train(
                model, 
                train_loader, 
                device, 
                optimizer, 
                criterion, 
                epoch, 
                l1_lambda=1e-5, 
                l2_lambda=1e-4
            )
            # Validate at every 10 epochs
            if epoch % 10 ==0 or epoch == 1:
                val_acc, val_prec, val_rec, val_f1_macro, val_f1_micro = evaluate(model, val_loader, device, label_names)
                # print(f"Validation Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
                print(f"Validation  Precision: {val_prec:.4f}  | Recall: {val_rec:.4f} | F1_macro: {val_f1_macro:.4f} | F1_micro: {val_f1_micro:.4f} ")

            # Collect metrics at the last epoch of each fold
            if epoch == NUM_EPOCHS:
                if epoch % 10 != 0 and epoch != 1:
                    val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device, label_names)
                    print(f"Final Evaluation - Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")
                all_precisions.append(val_prec)
                all_recalls.append(val_rec)
                all_f1s.append(val_f1_macro)

    # After all folds finished, compute and print statistics
    def print_stats(name, values):
        print(f"\n{name} stats across folds:")
        print(f"Mean:   {np.mean(values):.4f}")
        print(f"Std:    {np.std(values):.4f}")
        print(f"Min:    {np.min(values):.4f}")
        print(f"Median: {np.median(values):.4f}")
        print(f"Max:    {np.max(values):.4f}")

    print_stats("Precision", all_precisions)
    print_stats("Recall", all_recalls)
    print_stats("F1 Score", all_f1s)
if __name__ == "__main__":
    main()
