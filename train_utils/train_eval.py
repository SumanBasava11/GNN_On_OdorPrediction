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

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(loader):.4f} | "
          f"Train Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    return total_loss / len(loader), acc, prec, rec, f1

def evaluate(model, loader, device, valid_descriptors):   # output_threshold_file=None
    model.eval()
    all_preds, all_labels= [], []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            
            # To find optimal threshold per label
            logits = model(data) # raw outputs
            preds = torch.sigmoid(logits).cpu().numpy()                                              # preds = (torch.sigmoid(model(data)) > 0.4).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

    y_true = np.vstack(all_labels)
    y_probs = np.vstack(all_preds)
    y_preds = (y_probs> 0.4).astype(int)  
 
    acc = accuracy_score(y_true.flatten(), y_preds.flatten())
    f1 = f1_score(y_true, y_preds, average='macro', zero_division=1)
    prec = precision_score(y_true, y_preds, average='macro', zero_division=1)
    rec = recall_score(y_true, y_preds, average='macro', zero_division=1)

    log_confusion_matrices(y_true, y_preds, valid_descriptors)
    return acc, f1, prec, rec