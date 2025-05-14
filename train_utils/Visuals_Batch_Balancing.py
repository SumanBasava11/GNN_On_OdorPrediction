import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def save_label_distribution_bar_charts(
    dataloader,
    num_classes,
    save_dir="train_utils/batch_label_distributions",
    fold_num=0,
    max_batches=20
):
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        # Get labels
        if isinstance(batch, (tuple, list)):
            _, labels = batch
        else:
            labels = batch['labels']

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        # Sum label counts across batch
        label_counts = labels.sum(dim=0).cpu().numpy()

        # Plot bar chart
        plt.figure(figsize=(12, 4))
        plt.bar(np.arange(num_classes), label_counts)
        plt.xlabel("Label Index")
        plt.ylabel("Count in Batch")
        plt.title(f"Fold {fold_num}, Batch {batch_idx} - Label Distribution")
        plt.tight_layout()

        # Save chart
        file_name = f"fold_{fold_num}_batch_{batch_idx}.png"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path)
        plt.close()
