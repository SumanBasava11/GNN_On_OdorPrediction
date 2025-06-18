import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_topk_boxplot(per_label_scores, label_names, k=30, metric_name='F1 Score', output_file='topk_boxplot.png'):
    """
    Plots a boxplot of the top-k scoring labels based on mean score across folds.

    Args:
        per_label_scores (ndarray): Array of shape (n_folds, n_labels) with per-label metric scores.
        label_names (List[str]): Names of labels corresponding to columns in per_label_scores.
        k (int): Number of top labels to include based on mean score.
        metric_name (str): Name of the metric (e.g., 'F1 Score', 'AUROC').
        output_file (str): Path to save the resulting plot.
    """
    if isinstance(per_label_scores, list):
        per_label_scores = np.array(per_label_scores)

    assert per_label_scores.shape[1] == len(label_names), "Mismatch between scores and label names."

    mean_scores = np.mean(per_label_scores, axis=0)
    topk_indices = np.argsort(mean_scores)[-k:][::-1]

    topk_scores = per_label_scores[:, topk_indices]
    topk_labels = [label_names[i] for i in topk_indices]

    # Convert to long format DataFrame for seaborn
    df = pd.DataFrame(topk_scores, columns=topk_labels)
    df_melted = df.melt(var_name="Label", value_name=metric_name)

    plt.figure(figsize=(18, 7))
    sns.boxplot(x="Label", y=metric_name, data=df_melted, palette="coolwarm")
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top-{k} Labels by Mean {metric_name}")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved Top-{k} {metric_name} boxplot to {output_file}")

def plot_boxplots_per_label(label_names, all_f1s_per_label, all_aurocs_per_label, k=30):
    df_f1 = pd.DataFrame(np.array(all_f1s_per_label), columns=label_names)
    df_auroc = pd.DataFrame(np.array(all_aurocs_per_label), columns=label_names)

    plot_topk_boxplot(df_f1.values, label_names, k=k, metric_name='F1 Score', output_file=f"top{k}_boxplot_f1_score.png")
    plot_topk_boxplot(df_auroc.values, label_names, k=k, metric_name='AUROC', output_file=f"top{k}_boxplot_auroc.png")
