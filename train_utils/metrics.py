import os
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score

def save_per_label_metrics(model, loader, device, label_names, output_path="train_utils/per_label_metrics.txt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels)

    y_true = np.vstack(all_labels)
    y_pred = (np.vstack(all_preds) > 0.4).astype(int)

    precisions = precision_score(y_true, y_pred, average=None, zero_division=1)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=1)
    mean_precision = np.mean(precisions)

    with open(output_path, "w") as f:
        f.write("Label\tPrecision\tRecall\n")
        for i, label in enumerate(label_names):
            f.write(f"{label}\t{precisions[i]:.4f}\t{recalls[i]:.4f}\n")
        f.write(f"\nMean Precision: {mean_precision:.4f}\n")

    print(f"[Per-label metrics saved to {output_path}]")
    print(f"Mean Precision over all labels: {mean_precision:.4f}")
