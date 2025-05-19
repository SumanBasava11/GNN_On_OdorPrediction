import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_label_distribution_per_fold(fold_idx, labels, split_name, label_names, save_dir="fold_distributions"):

    os.makedirs(save_dir, exist_ok=True)

    pos_counts = np.sum(labels, axis=0)
    total_molecules = labels.shape[0]
    total_positive_labels = int(pos_counts.sum())

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(label_names)), pos_counts, color='skyblue')
    plt.xticks(range(len(label_names)), label_names, rotation='vertical', fontsize=6)
    plt.ylabel("Positive Samples")

    plt.title(
        f"{split_name} Label Distribution - Fold {fold_idx}\n"
        f"Total Molecules: {total_molecules} | Total Positive Labels: {total_positive_labels}"
    )

    plt.tight_layout()
    filename = f"{split_name.lower()}_fold{fold_idx}_distribution.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
