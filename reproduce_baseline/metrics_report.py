import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score

def write_metrics_report(y_true, y_pred_probs, y_pred_labels, label_names, filename="metrics_report.txt"):
    """
    Writes per-label AUC-ROC and Precision sorted by descending AUC-ROC to a txt file.
    
    Args:
        y_true (np.ndarray): True binary labels, shape (num_samples, num_labels).
        y_pred_probs (np.ndarray): Predicted probabilities, shape (num_samples, num_labels).
        y_pred_labels (np.ndarray): Predicted binary labels, shape (num_samples, num_labels).
        label_names (list): List of label names.
        filename (str): Output filename.
    """
    aucs = []
    precisions = []
    
    for i, label in enumerate(label_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            # Handle case where only one class present in y_true[:, i]
            auc = float('nan')
        prec = precision_score(y_true[:, i], y_pred_labels[:, i], zero_division=0)
        aucs.append(auc)
        precisions.append(prec)
    
    df = pd.DataFrame({
        "Label": label_names,
        "AUC-ROC": aucs,
        "Precision": precisions
    })
    
    df = df.sort_values(by="AUC-ROC", ascending=False)
    
    with open(filename, "w") as f:
        f.write(f"{'Label':<30}\t{'AUC-ROC':<10}\t{'Precision':<10}\n")
        for _, row in df.iterrows():
            f.write(f"{row['Label']:<30}\t{row['AUC-ROC']:.4f}\t{row['Precision']:.4f}\n")
