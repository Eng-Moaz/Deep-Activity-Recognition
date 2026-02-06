import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from helper_utils.evaluation import evaluate_test_set
from data_utils.dataloader import get_data_loaders
from model import Baseline4


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train():

    cfg = load_config("config.yaml")
    device = torch.device(cfg['system']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_data_loaders(cfg, mode="temporal")

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")

    print(f"Loading Backbone from: {cfg['paths']['trained_resnet']}")
    model = Baseline4(
        trained_resnet_path=cfg['paths']['trained_resnet'],
        hidden_size=cfg['model']['hidden_size'],
        lstm_layers=cfg['model']['lstm_layers'],
        num_classes=cfg['model']['num_classes']
    )
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )

    scheduler = None
    if cfg['scheduler']['use_scheduler']:
        scheduler = StepLR(
            optimizer,
            step_size=cfg['scheduler']['step_size'],
            gamma=cfg['scheduler']['gamma']
        )

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    epochs = cfg['training']['epochs']

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        #Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc="Training")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        train_acc = 100 * correct / total
        print(f"    Train Loss: {running_loss / len(train_loader):.4f} | Acc: {train_acc:.2f}%")

        #Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"    Val Acc: {val_acc:.2f}%")

        #Model saving
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg['paths']['model_save_path'])
            print(f"Saved Best Model ({best_acc:.2f}%)")

        if scheduler:
            scheduler.step()

    model.load_state_dict(torch.load(cfg['paths']['model_save_path'], map_location=device))
    test_acc = evaluate_test_set(model, test_loader, device, cfg["model"]["class_names"],cfg['paths']['cm_save_path'])
    print(f"FINAL TEST ACCURACY: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    train()
