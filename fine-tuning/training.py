import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train_epoch(model,
                data_loader, 
                optimizer,
                criterion, 
                device):
    """
    roda uma época de treino
    retorna:
    - avg_loss: training loss média over all batches
    - train_acc: training accuracy over all batches
    """
    model.to(device)
    model.train()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        loss = criterion(logits, labels) # era loss_fn

        total_loss += loss.item()  
        loss.backward()
        optimizer.step()

        # Collect predictions for this batch
        preds = torch.argmax(logits, dim=1).flatten()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    # Compute training accuracy
    epoch_acc = accuracy_score(np.array(all_labels), np.array(all_preds))
    return avg_loss, epoch_acc

def compute_metrics(preds, labels):
    """
    preds: np.array das predições (0 ou 1), shape (N,)
    labels: np.array das labels de vdd, shape (N,)
    
    retorna um dict com:
    - accuracy
    - precision (pra classe 1)
    - recall (pra classe 1)
    - f1 (pra classe 1)
    """
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", pos_label=1
    )
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
    
def eval_model(model, 
            data_loader,
            criterion,
            device):
    """
    val loop
    rretorna:
    - avg_loss: average validation loss
    - val_acc: validation accuracy
    - metrics: dict containing {accuracy, precision, recall, f1}
    """
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).flatten()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    val_acc = metrics["accuracy"]
    return avg_loss, val_acc, metrics