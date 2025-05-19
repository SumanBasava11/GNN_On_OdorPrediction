import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from train_utils.utils import log_confusion_matrices

def train(model, loader, device, optimizer, criterion, epoch, l1_lambda=1e-5, l2_lambda=1e-4):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)

        # Apply L1/L2 regularization only to MLP parameters
        # mlp_params = [param for name, param in model.named_parameters() if "mlp" in name and param.requires_grad]
        
        reg_params = [param for name, param in model.named_parameters() 
              if ("mlp" in name or "gcn" in name) and param.requires_grad]

        l1_norm = sum(p.abs().sum() for p in  reg_params)
        l2_norm = sum(p.pow(2).sum() for p in  reg_params)
        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        preds = (torch.sigmoid(output) > 0.3).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    train_acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    train_prec = precision_score(y_true, y_pred, average='macro', zero_division=1)
    train_rec = recall_score(y_true, y_pred, average='macro', zero_division=1)
    train_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=1)
    train_f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=1)

    # print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(loader):.4f} | "
    #       f"Train Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    print(f"Epoch {epoch:03d} | Train | Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1_macro: {train_f1_macro:.4f} | F1_micro: {train_f1_micro:.4f}")
    # print("Positive predictions ratio:", preds.mean())

    return total_loss / len(loader), train_acc, train_prec, train_rec, train_f1_macro, train_f1_micro

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

    # # Step 1: Tune threshold per label using precision-recall curve
    # best_thresholds = []
    # for i in range(y_true.shape[1]):
    #     try:
    #         precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
    #         f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    #         best_threshold = thresholds[np.argmax(f1_scores)]
    #     except ValueError:
    #         best_threshold = 0.5  # fallback if only one class present
    #     best_thresholds.append(best_threshold)

    # best_thresholds = np.array(best_thresholds)

    # # Step 2: Apply thresholds
    # y_preds = (y_probs >= best_thresholds).astype(int)

    y_preds = (y_probs> 0.3).astype(int)  
 
    val_acc = accuracy_score(y_true.flatten(), y_preds.flatten())
    val_f1_macro = f1_score(y_true, y_preds, average='macro', zero_division=1)
    val_f1_micro = f1_score(y_true, y_preds, average='micro', zero_division=1)
    val_prec = precision_score(y_true, y_preds, average='macro', zero_division=1)
    val_rec = recall_score(y_true, y_preds, average='macro', zero_division=1)

    log_confusion_matrices(y_true, y_preds, valid_descriptors)
    return val_acc, val_prec, val_rec, val_f1_macro, val_f1_micro