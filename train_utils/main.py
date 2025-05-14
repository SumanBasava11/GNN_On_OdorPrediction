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
# from train_utils.BatchBalancer import IterativeStratifiedBatchSampler
from train_utils.Visuals_Batch_Balancing import *
from sklearn.metrics import roc_auc_score

# Suppress RDKit warnings
rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.ERROR)

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

    with open(output_path, "w") as f:
        f.write("Label\tAUC-ROC\n")
        for i, name in enumerate(label_names):
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            except ValueError:
                auc = float('nan')
            f.write(f"{name}\t{auc:.4f}\n")

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

    for fold, (train_idx, val_idx) in enumerate(mskf.split(smiles, labels), 1):
        print(f"\nFold {fold}/{N_SPLITS} {'=' * 40}")

        # Model initialization
        model = OdorClassifier(num_tasks=labels.shape[1], readout_dim=175, mlp_dims=[96, 63]).to(device)
        
        train_loader = DataLoader(OdorDataset(smiles[train_idx], labels[train_idx]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(OdorDataset(smiles[val_idx], labels[val_idx]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        print("\nModel Summary:\n")
        print(model)
       
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # Choose loss function
        if USE_FOCAL:
            criterion = lambda out, tgt: sigmoid_focal_loss(out, tgt, alpha=0.5, gamma=2, reduction="mean")
        else:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        
        # # Train dataset
        # train_dataset = OdorDataset(smiles[train_idx], labels[train_idx])
        # # # Use StratifiedBatchSampler instead of default sampler
        # # train_sampler = IterativeStratifiedBatchSampler(labels[train_idx], batch_size=64)
        # # Define train loader with stratified sampling
        # train_loader = DataLoader(
        #     train_dataset, 
        #     batch_size=BATCH_SIZE, 
        #     sampler=train_sampler,
        #     collate_fn=collate_fn
        # )
       
        save_label_distribution_bar_charts(train_loader, num_classes=labels.shape[1], save_dir="plots", fold_num=0, max_batches=20)

        # # Validation dataset and loader
        # val_loader = DataLoader(
        #     OdorDataset(smiles[val_idx], labels[val_idx]), 
        #     batch_size=BATCH_SIZE, 
        #     shuffle=False,  # No need to shuffle for validation
        #     collate_fn=collate_fn
        # )

        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Number of batches in train_loader: {len(train_loader)}")

        print(f"Validation set size: {len(val_loader.dataset)}")
        print(f"Number of batches in val_loader: {len(val_loader)}")
        
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
                
                compute_auc_per_label(model, val_loader, device, labels_df.columns.tolist(), output_path=f"auc-roc/auc_roc_fold{fold}_epoch{epoch}.txt")
                save_per_label_metrics(
                    model=model,
                    loader=train_loader,
                    device=device,
                    label_names=label_names,
                    output_path=f"metrics/per_label_metrics_fold{fold}_epoch{epoch}.txt"
                )

if __name__ == "__main__":
    main()
