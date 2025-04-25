import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
        preds = (torch.sigmoid(output) > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    acc = accuracy_score(np.concatenate(all_labels).flatten(), np.concatenate(all_preds).flatten())
    print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(loader):.4f} | Train Acc: {acc:.4f}")
    return total_loss / len(loader), acc

def evaluate(model, loader, device, valid_descriptors):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            preds = (torch.sigmoid(model(data)) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=1)

    log_confusion_matrices(y_true, y_pred, valid_descriptors)
    return acc, f1, prec, rec