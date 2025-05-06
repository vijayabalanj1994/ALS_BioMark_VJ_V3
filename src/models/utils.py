import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from contourpy import contour_generator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

from config.config import config
from src.process_data.load_data import get_dataloaders_and_classweights
from src.models.DenseNet121 import AttentionDenseNet121

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc

def validate_one_epoch(model, val_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc

def train_loop(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, num_epoch):

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    best_val_acc = -1

    print("\ttraining the model.")
    for epoch in range(num_epoch):
        print(f"\tEpoch: {epoch+1}/{num_epoch}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss , val_acc = validate_one_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)


        # string the best performing model.
        model_path = os.path.join(config.saved_models_dir_path,f"fold_{config.fold_no}_model_weights.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),model_path)

        print(f"\t\tTrainLoss: {train_loss:.4f}, TrainAccuracy: {train_acc:.4f} ValLoss: {val_loss:.4f}, ValAccuracy: {val_acc:.4f}")

    # storing the results in file
    results_path = os.path.join(config.logs_dir_path, f"fold_{config.fold_no}_training_results.csv")
    training_results = pd.DataFrame({
        "epoch": list(range(1,num_epoch+1)),
        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "val_loss": val_loss_list,
        "val_acc": val_acc_list
    })
    training_results.to_csv(results_path, index=False)

def sensitivity_specificity(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    num_classes = cm.shape[0]
    sensitivities = []
    specificities = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return sensitivities, specificities

def evaluate_model(fold, val_loader):
    print("\tEvaluating the model")

    model_path = os.path.join(config.saved_models_dir_path, f"fold_{config.fold_no}_model_weights.pth")
    model = AttentionDenseNet121().to(config.device)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    sens, spec = sensitivity_specificity(y_true, y_pred, labels=[0,1,2])

    #saving evaluation results
    results_path = os.path.join(config.logs_dir_path, f"fold_{config.fold_no}_evaluation_results.csv")

    # Create a DataFrame for saving
    results_df = pd.DataFrame({
        'Class': [0,1,2],
        'Sensitivity': sens,
        'Specificity': spec
    })

    # Add overall metrics
    overall_metrics = pd.DataFrame({
        'Class': ['Overall'],
        'Sensitivity': [np.nan],
        'Specificity': [np.nan],
        'Accuracy': [acc],
        'MCC': [mcc]
    })

    results_df = pd.concat([results_df, overall_metrics], ignore_index=True)
    results_df.to_csv(results_path, index=False)

def run_folds(fold):

    train_loader, val_loader, class_weights =  get_dataloaders_and_classweights(fold=fold)
    class_weights = class_weights.to(config.device)

    model = AttentionDenseNet121().to(config.device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_loop(model, train_loader, val_loader, optimizer, scheduler, loss_fn, config.device, config.no_of_epoch)
    evaluate_model(fold, val_loader)

