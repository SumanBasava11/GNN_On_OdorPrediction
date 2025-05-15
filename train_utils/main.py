import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# from torchvision.ops import sigmoid_focal_loss
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
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress RDKit warnings
rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.ERROR)

# def visualize_label_distribution_per_fold(fold_idx, labels, split_name, label_names, save_dir="fold_distributions"):
#     os.makedirs(save_dir, exist_ok=True)
#     pos_counts = np.sum(labels, axis=0)

#     plt.figure(figsize=(12, 5))
#     plt.bar(range(len(label_names)), pos_counts, color='skyblue')
#     plt.xticks(range(len(label_names)), label_names, rotation='vertical', fontsize=6)
#     plt.ylabel("Positive Samples")
#     plt.title(f"{split_name} Label Distribution - Fold {fold_idx}")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f"{split_name.lower()}_fold{fold_idx}_distribution.png"))
#     plt.close()

def focal_loss(logits, targets, gamma=2.0, reduction='mean', eps=1e-6):
    """
    Args:
        logits: Raw model outputs (before sigmoid), shape [batch_size, num_classes]
        targets: Binary labels (0 or 1), same shape
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
    """
    # Apply sigmoid to get predicted probabilities
    p = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)  # Ensure p is in the range [eps, 1-eps]
    
    # Compute the cross-entropy components for each label
    loss_pos = targets * (1 - p) ** gamma * torch.log(p)  # Term for positive samples
    loss_neg = (1 - targets) * p ** gamma * torch.log(1 - p)  # Term for negative samples
    
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

    # with open(output_path, "w") as f:
    #     f.write("Label\tAUC-ROC\n")
    #     for i, name in enumerate(label_names):
    #         try:
    #             auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
    #         except ValueError:
    #             auc = float('nan')
    #         f.write(f"{name}\t{auc:.4f}\n")

def main():
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/Balanced_OdorSmiles_Top30.csv', encoding='ISO-8859-1')
    
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

    for fold, (train_idx, val_idx) in enumerate(mskf.split(smiles, labels), 1):
        print(f"\nFold {fold}/{N_SPLITS} {'=' * 40}")

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        # # Visualize per-fold label distribution
        # visualize_label_distribution_per_fold(fold, train_labels, "Train", label_names)
        # visualize_label_distribution_per_fold(fold, val_labels, "Validation", label_names)

        # Model initialization
        model = OdorClassifier(num_tasks=labels.shape[1], readout_dim=175, mlp_dims=[96, 63]).to(device)
        
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
        
        # # --- Collect batch-wise label distribution for first 50 batches ---
        # batch_label_counts = []
        # for i, (data, labels_batch) in enumerate(train_loader):
        #     batch_counts = labels_batch.sum(dim=0).numpy()
        #     batch_label_counts.append(batch_counts)

        #    # Convert to DataFrame for plotting
        #     batch_df = pd.DataFrame(batch_label_counts, columns=label_names)
        #     batch_df["Batch"] = batch_df.index + 1

        #     print(f"\n[Batch-wise Label Distribution - Batch {i + 1} - Fold {fold}]")
        #     for j, row in batch_df.iterrows():
        #         print(f"Batch {i + 1:02d}: ", {name: int(row[name]) for name in label_names})

        #     # --- Boxplot visualization per batch ---
        #     melted = batch_df.melt(id_vars=["Batch"], var_name="Label", value_name="Positive Samples")
        #     plt.figure(figsize=(12, 5))
        #     sns.boxplot(data=melted, x="Label", y="Positive Samples")
        #     plt.xticks(rotation=90)
        #     plt.title(f"Label-wise Distribution in Batch {i + 1} (Fold {fold})")
        #     plt.tight_layout()

        #     # Save the plot for each batch
        #     plot_path = f"batchwise_distribution_fold{fold}_batch{i + 1}.png"
        #     plt.savefig(plot_path)
        #     plt.close()  # Close the plot to avoid memory overflow
        #     print(f"Boxplot saved to: {plot_path}")

        # Start training loop
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc, train_prec, train_rec, train_f1 = train(
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
            if epoch % 10 or epoch == 1:
                acc, prec, rec, f1 = evaluate(model, val_loader, device, label_names)
                print(f"Validation Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
                
                # compute_auc_per_label(model, val_loader, device, labels_df.columns.tolist(), output_path=f"auc-roc/auc_roc_fold{fold}_epoch{epoch}.txt")
                # save_per_label_metrics(
                #     model=model,
                #     loader=train_loader,
                #     device=device,
                #     label_names=label_names,
                #     output_path=f"metrics/per_label_metrics_fold{fold}_epoch{epoch}.txt"
                # )

if __name__ == "__main__":
    main()
