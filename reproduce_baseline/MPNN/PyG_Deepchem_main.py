import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import deepchem as dc
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from reproduce_baseline.MPNN.deepchem_featuriser import GraphFeaturizer
from reproduce_baseline.MPNN.PyTorch_deepchem import DeepChemGraphClassifier 

from sklearn.preprocessing import MultiLabelBinarizer

# Hyperparameters
batch_size = 100
n_epochs = 30
n_folds = 5
input_dim = 111
n_tasks = 138
hidden_dim = 128
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def focal_loss(inputs, targets, gamma=2.0, alpha=0.25):
    BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
    pt = torch.where(targets == 1, inputs, 1-inputs)
    loss = alpha * (1-pt)**gamma * BCE_loss
    return loss.mean()

def compute_metrics(y_true, y_pred):
    metric_auroc = dc.metrics.Metric(
        dc.metrics.roc_auc_score, average="macro", task_averager=np.mean, mode="classification"
    )
    metric_f1 = dc.metrics.Metric(
        dc.metrics.f1_score, average="macro", task_averager=np.mean, mode="classification"
    )
    metric_prec = dc.metrics.Metric(
        dc.metrics.precision_score, average="macro", task_averager=np.mean, mode="classification"
    )
    metric_rec = dc.metrics.Metric(
        dc.metrics.recall_score, average="macro", task_averager=np.mean, mode="classification"
    )

    scores = {
        "AUROC": metric_auroc.compute_metric(y_true, y_pred)[0],
        "F1": metric_f1.compute_metric(y_true, y_pred)[0],
        "Precision": metric_prec.compute_metric(y_true, y_pred)[0],
        "Recall": metric_rec.compute_metric(y_true, y_pred)[0],
    }
    return scores

def batch_iter(X, y, batch_size=32, shuffle=True):
    idxs = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idxs)
    for start_idx in range(0, len(X), batch_size):
        batch_idxs = idxs[start_idx:start_idx + batch_size]
        yield [X[i] for i in batch_idxs], y[batch_idxs]


def train_one_epoch(model, dataset, optimizer):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in tqdm(batch_iter(dataset.X, dataset.y, batch_size=batch_size), desc="Training"):
        # X_batch: list of PyG Data objects
        # y_batch: numpy array
        x = [d.x for d in X_batch]
        edge_index = [d.edge_index for d in X_batch]
        edge_attr = [d.edge_attr for d in X_batch]
        labels = torch.tensor(y_batch, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model([x, edge_index, edge_attr])
        preds = outputs[:, :, 1]
        loss = focal_loss(preds, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss

def evaluate(model, dataset):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(batch_iter(dataset.X, dataset.y, batch_size=batch_size), desc="Evaluating"):
            # X_batch: list of PyG Data objects
            # y_batch: numpy array
            x = [d.x for d in X_batch]
            edge_index = [d.edge_index for d in X_batch]
            edge_attr = [d.edge_attr for d in X_batch]
            labels = torch.tensor(y_batch, dtype=torch.float32).to(device)

            outputs = model([x, edge_index, edge_attr])
            preds = outputs[:, :, 1]  # prob of class 1

            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    return compute_metrics(y_true, y_pred)

def main():

    # Path to your dataset
    DATASET = 'C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/Data_Sampling/FrequentOdor_extraction/(sat)mapped+unmapped_odors_openPOM_Top138.csv'
    smiles_field = 'smiles'

    df = pd.read_csv(DATASET)

    TASKS = df.columns[2:].tolist()
    print(f"Identified {len(TASKS)} tasks.")
    # print("Task names:", TASKS)

    # num_molecules = df.shape[0]
    # print("Number of molecules in dataset:", num_molecules)

    featurizer = GraphFeaturizer()

    loader = dc.data.CSVLoader(tasks=TASKS,
                   feature_field=smiles_field,
                   featurizer=featurizer)

    dc_dataset = loader.featurize(DATASET)
    print("Featurization complete.")

    # Assuming df.X is list of Data objects, df.y is numpy array
    X, y = dc_dataset.X, dc_dataset.y

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)

    folds = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"\n=== Fold {fold+1}/{n_folds} ===")

        train_X, val_X = [X[i] for i in train_idx], [X[i] for i in val_idx]
        train_y, val_y = y[train_idx], y[val_idx]

        train_dataset = dc.data.NumpyDataset(train_X, train_y)
        val_dataset = dc.data.NumpyDataset(val_X, val_y)

        model = DeepChemGraphClassifier(batch_size=batch_size, input_dim=input_dim, n_tasks=n_tasks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_dataset, optimizer)
            metrics = evaluate(model, val_dataset)
            print(
                f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f} "
                f"AUROC: {metrics['AUROC']:.4f} F1: {metrics['F1']:.4f} "
                f"Precision: {metrics['Precision']:.4f} Recall: {metrics['Recall']:.4f}"
            )

        all_metrics.append(metrics)

    # Mean over folds
    mean_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print("\n=== Cross-validated Macro-Averaged Metrics ===")
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()