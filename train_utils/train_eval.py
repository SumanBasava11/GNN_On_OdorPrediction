import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from train_utils.utils import log_confusion_matrices

def train(model, loader, device, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        data, labels = batch
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(output) > 0.4).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    acc = accuracy_score(np.concatenate(all_labels).flatten(), np.concatenate(all_preds).flatten())
    print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(loader):.4f} | Train Acc: {acc:.4f}")
    return total_loss / len(loader), acc

def evaluate(model, loader, device, valid_descriptors, output_threshold_file=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            
            # To find optimal threshold per label
            logits = model(data) # raw outputs
            preds = torch.sigmoid(logits).cpu().numpy()                                              # preds = (torch.sigmoid(model(data)) > 0.4).cpu().numpy()
           
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_probs = np.vstack(all_preds)

    # Find optimal thresholds for each label based on precision-recall curve
    optimal_thresholds = []
    for i in range(y_true.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid division by zero
        optimal_threshold = thresholds[np.argmax(f1_scores)]  # Choose the threshold with the best F1 score
        optimal_thresholds.append(optimal_threshold)

    # Save the optimal thresholds per label in a text file
    with open(output_threshold_file, "w") as f:
        f.write("Label\tOptimal Threshold\n")
        for i, threshold in enumerate(optimal_thresholds):
            label_name = valid_descriptors[i] if i < len(valid_descriptors) else f"Label_{i}"
            f.write(f"{label_name}\t{threshold:.4f}\n")

    # Apply per-label thresholds to make final predictions
    final_preds = (y_probs > optimal_thresholds).astype(int)

    acc = accuracy_score(y_true.flatten(), final_preds.flatten())
    f1 = f1_score(y_true, final_preds, average='macro', zero_division=1)
    prec = precision_score(y_true, final_preds, average='macro', zero_division=1)
    rec = recall_score(y_true, final_preds, average='macro', zero_division=1)

    log_confusion_matrices(y_true, final_preds, valid_descriptors)
    return acc, f1, prec, rec