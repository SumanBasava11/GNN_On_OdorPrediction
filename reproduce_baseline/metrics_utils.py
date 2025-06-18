import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from scipy import stats

def positive_class_accuracy(y_true, y_pred):
    # Only consider samples where label=1 for accuracy (correct positive predictions / total positives)
    positive_indices = (y_true == 1)
    if np.sum(positive_indices) == 0:
        return float('nan')  # Avoid div by zero if no positives
    correct_pos = (y_true[positive_indices] == y_pred[positive_indices]).sum()
    return correct_pos / positive_indices.sum()


def compute_labelwise_mean_metrics(y_true, y_pred, y_prob, label_names):
    per_label_prec = []
    per_label_rec = []
    per_label_f1 = []
    per_label_acc = []
    per_label_auroc = []

    for i, label in enumerate(label_names):
        y_true_label = y_true[:, i]
        y_pred_label = y_pred[:, i]
        y_prob_label = y_prob[:, i]

        if np.sum(y_true_label) == 0:
            continue

        prec = precision_score(y_true_label, y_pred_label, zero_division=1)
        rec = recall_score(y_true_label, y_pred_label, zero_division=1)
        f1 = f1_score(y_true_label, y_pred_label, zero_division=1)
        acc = accuracy_score(y_true_label, y_pred_label)
        try:
            auroc = roc_auc_score(y_true_label, y_prob_label)
        except ValueError:
            auroc = float('nan')

        per_label_prec.append(prec)
        per_label_rec.append(rec)
        per_label_f1.append(f1)
        per_label_acc.append(acc)
        per_label_auroc.append(auroc)

    mean_prec = np.nanmean(per_label_prec)
    mean_rec = np.nanmean(per_label_rec)
    mean_f1 = np.nanmean(per_label_f1)
    mean_acc = np.nanmean(per_label_acc)
    mean_auroc = np.nanmean(per_label_auroc)

    return mean_prec, mean_rec, mean_f1, mean_acc, mean_auroc

def compute_confidence_interval(values, confidence=0.95):
    values = np.array(values)
    mean = np.nanmean(values)
    sem = stats.sem(values, nan_policy="omit")
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, len(values) - 1)
    return mean, mean - margin, mean + margin
