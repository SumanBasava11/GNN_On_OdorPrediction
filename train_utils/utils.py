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