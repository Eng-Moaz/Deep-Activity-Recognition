import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config
from model import Baseline4
from data_utils.dataloader import get_data_loaders
from helper_utils.evaluation import evaluate_test_set


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)
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

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train():
    # Instantiate Config
    cfg = Config()

    # Setup Device
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment: {cfg.experiment_name}")

    # Data Loaders
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # Model Setup
    print(f"Loading Backbone from: {cfg.trained_resnet_path}")
    model = Baseline4(
        trained_resnet_path=cfg.trained_resnet_path,
        hidden_size=cfg.hidden_size,
        lstm_layers=cfg.lstm_layers,
        num_classes=cfg.num_classes
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # Scheduler
    scheduler = None
    if cfg.use_scheduler:
        scheduler = StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma
        )

    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_acc = 0.0

    # Create save directory
    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"    Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"    Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"    Saved Best Model ({best_acc:.2f}%)")

        if scheduler:
            scheduler.step()

    # Final Evaluation
    print("\nEvaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))

    test_acc = evaluate_test_set(
        model,
        test_loader,
        device,
        cfg.class_names,
        cfg.cm_save_path
    )
    print(f"FINAL TEST ACCURACY: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    train()