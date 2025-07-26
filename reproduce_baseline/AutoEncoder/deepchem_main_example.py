import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepchem as dc
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from reproduce_baseline.configuration import *
from reproduce_baseline.AutoEncoder.UNetModel import OdorGCNUNet
from reproduce_baseline.Dataset import OdorDataset, collate_fn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from deepchem.metrics import Metric
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reconstruction_loss(node_out, x):
    # Example reconstruction loss: MSE between node features and predicted node output
    return F.mse_loss(node_out, x)

# ====== Define Focal Loss ======
def focal_loss(logits, targets, alpha=1.0, gamma=2.0, reduction='mean'):
    """Focal Loss for multi-label classification."""
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_term = (1 - p_t) ** gamma
    loss = alpha * focal_term * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# ======= WRAP WITH DEEPCHEM TORCHMODEL ========
class DeepChemOdorModel(dc.models.TorchModel):
    def __init__(self, pyg_model, alpha=0.1, **kwargs):

        # Define loss function
        def loss_function(outputs, labels, weights=None):
            graph_out, node_out = outputs
            labels = labels.float().to(graph_out.device)
            graph_loss = focal_loss(graph_out, labels, alpha=0.5, gamma=2.0)
            return graph_loss
        
        super().__init__(model=pyg_model, loss=loss_function, **kwargs)
        self.alpha = alpha

    def default_predict(self, inputs):
        graph_out, node_out = self.model(inputs[0])
        return graph_out

def make_dc_dataset(odor_dataset, model):
    embeddings = []
    labels_list = []

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data, labels in odor_dataset:
            data = data.to(device)
            label = labels.to(device)

            # Forward pass to get graph-level embedding
            graph_out, _ = model(data)

            # Store results
            embeddings.append(graph_out.cpu().numpy())
            labels_list.append(label.cpu().numpy())

    # Stack to create feature matrix and label matrix
    X = np.vstack(embeddings)
    y = np.vstack(labels_list)

    return dc.data.NumpyDataset(X, y)

def main():
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv', encoding='ISO-8859-1')  # Update this path accordingly
    smiles_list = df['smiles'].tolist()
    labels = df.drop(['smiles', 'descriptors'], axis=1).values
    num_tasks = labels.shape[1]

    # ======= METRICS ========
    metrics = [
        Metric(roc_auc_score, mode="classification", name="AUROC"),
        Metric(precision_score, mode="classification", name="Precision"),
        Metric(recall_score, mode="classification", name="Recall"),
        Metric(f1_score, mode="classification", name="F1")
    ]

    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_fold_train_metrics = []
    all_fold_val_metrics = []

    for fold, (train_idx, val_idx) in enumerate(mskf.split(smiles_list, labels)):
        print(f"\n===== Fold {fold + 1} / {N_SPLITS} =====")
        
        # Prepare datasets
        train_dataset_raw = OdorDataset([smiles_list[i] for i in train_idx], labels[train_idx])
        val_dataset_raw = OdorDataset([smiles_list[i] for i in val_idx], labels[val_idx])

        print("\n[DEBUG] Inspecting one sample from train_dataset_raw:")
        sample_data, sample_label = train_dataset_raw[0]
        print("Sample graph:", sample_data)
        print("Sample label:", sample_label)
        print("=========================\n")

        # Initialize model
        pyg_model = OdorGCNUNet(num_tasks=num_tasks).to(device)

        # Convert to DeepChem NumpyDataset using embeddings
        print("Encoding training dataset...")
        train_dc_dataset = make_dc_dataset(train_dataset_raw, pyg_model)

        print("Encoding validation dataset...")
        val_dc_dataset = make_dc_dataset(val_dataset_raw, pyg_model)

        # Initialize DeepChem model
        model = DeepChemOdorModel(pyg_model, alpha=ALPHA, model_dir=f'./model_fold_{fold}')

        # Train the model
        print("Training...")
        model.fit(train_dc_dataset, nb_epoch=NUM_EPOCHS)

        # Evaluate on train and val using DeepChem’s predict + metrics
        def evaluate(dataset):
            y_pred_logits = model.predict(dataset)
            y_pred_probs = torch.sigmoid(torch.from_numpy(y_pred_logits)).numpy()
            y_true = dataset.y
            scores = {m.name: m.compute_metric(y_true, y_pred_probs) for m in metrics}
            return scores

        train_scores = evaluate(train_dc_dataset)
        val_scores = evaluate(val_dc_dataset)

        print("Train Metrics:", {k: f"{v:.4f}" for k, v in train_scores.items()})
        print("Val Metrics:  ", {k: f"{v:.4f}" for k, v in val_scores.items()})

        all_fold_train_metrics.append(train_scores)
        all_fold_val_metrics.append(val_scores)

    # Compute Mean Metrics over Folds
    def mean_metrics(metric_list):
        return {key: np.mean([m[key] for m in metric_list]) for key in metric_list[0]}

    print("\n====== Summary Across All Folds ======")
    print("Mean Train Metrics:")
    for k, v in mean_metrics(all_fold_train_metrics).items():
        print(f"{k}: {v:.4f}")
    print("\nMean Validation Metrics:")
    for k, v in mean_metrics(all_fold_val_metrics).items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()