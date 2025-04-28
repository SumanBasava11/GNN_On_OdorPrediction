import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def log_confusion_matrices(true, pred, labels):
    label_accuracy = (true == pred).sum(axis=0) / true.shape[0]
    best = np.argsort(label_accuracy)[-20:]
    worst = np.argsort(label_accuracy)[:20]

    # print("\nTop 20 Best Predicted Labels:")
    # for idx in best:
    #     print(f"{labels[idx]}: Accuracy {label_accuracy[idx]:.4f}")

    # print("\nTop 20 Worst Predicted Labels:")
    # for idx in worst:
    #     print(f"{labels[idx]}: Accuracy {label_accuracy[idx]:.4f}")

    plot_conf_matrix(confusion_matrix(true.flatten(), pred.flatten()), "All Labels", "total_cm.png")
    plot_conf_matrix(confusion_matrix(true[:, best].flatten(), pred[:, best].flatten()), "Best Labels", "best_cm.png")
    plot_conf_matrix(confusion_matrix(true[:, worst].flatten(), pred[:, worst].flatten()), "Worst Labels", "worst_cm.png")

def plot_conf_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

import os

def save_label_distribution(labels, valid_descriptors, path="C:/Users/suman/OneDrive/Bureau/Internship_Study/GNN_On_OdorPrediction/train_utils/label_distribution.txt"):
    label_counts = labels.sum(axis=0)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        f.write("[INFO] Label Distribution:\n\n")
        for label, count in zip(valid_descriptors, label_counts):
            line = f"{label}: {int(count)} samples"
            print(line)
            f.write(line + "\n")
