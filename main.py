import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torchvision.ops import sigmoid_focal_loss
import warnings
from rdkit import RDLogger
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter("ignore", category=UndefinedMetricWarning)

from config import *
from GNN_Model.gcn_model import OdorClassifier
from train_utils.dataset import OdorDataset, collate_fn
from train_utils.train_eval import train, evaluate

# Suppress RDKit warnings
rd_logger = RDLogger.logger()
rd_logger.setLevel(RDLogger.ERROR)

def main():
    df = pd.read_csv('C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/data/OdorSmiles_Updated.csv', encoding='ISO-8859-1')
    smiles = df["SMILES"].values
    labels_df = df.drop(columns=["SMILES", "cas_number"])
    valid_descriptors = labels_df.loc[:, labels_df.sum() > 10].columns
    labels = labels_df[valid_descriptors].values

    pos_counts = (labels == 1).sum(axis=0)
    neg_counts = (labels == 0).sum(axis=0)
    pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(mskf.split(smiles, labels), 1):
        print(f"\nFold {fold}/{N_SPLITS} {'=' * 40}")
        model = OdorClassifier(num_tasks=labels.shape[1], readout_dim=175, mlp_dims=[96, 63]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        if USE_FOCAL:
            criterion = lambda out, tgt: sigmoid_focal_loss(out, tgt, alpha=0.25, gamma=0.75, reduction="mean")
        else:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

        train_loader = DataLoader(OdorDataset(smiles[train_idx], labels[train_idx]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(OdorDataset(smiles[val_idx], labels[val_idx]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        for epoch in range(1, NUM_EPOCHS + 1):
            train(model, train_loader, device, optimizer, criterion, epoch)
            if epoch == 1 or epoch % 10 == 0:
                acc, f1, prec, rec = evaluate(model, val_loader, device, valid_descriptors)
                print(f"Validation Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

if __name__ == "__main__":
    main()
