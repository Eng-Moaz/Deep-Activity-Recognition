import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import Config_stg1 , Config_stg2
from model import Baseline3_stg1 , Baseline3_stg2
from data_utils.dataloader import get_data_loaders
from helper_utils.evaluation import evaluate_test_set


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)

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

        # Update progress bar
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
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_stg(stage):
    # Instantiate Config
    cfg = Config_stg1() if stage == 1 else Config_stg2()

    # Setup Device
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment: {cfg.experiment_name}")

    # Data Loaders
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # Model Setup
    if stage == 1:
        model = Baseline3_stg1(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout
        ).to(device)
    else:
        # Stage 2 requires loading the saved weights from Stage 1
        model = Baseline3_stg2(
            saved_resnet_path=cfg.saved_resnet50_path,
            num_classes=cfg.num_classes
        ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_acc = 0.0

    # Create save directory if it doesn't exist
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

    # 7. Final Evaluation
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
    parser = argparse.ArgumentParser(description="Train Baseline 3 Stages")

    # Add argument for stage (1 or 2)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="Which stage to train: 1 for Player Actions, 2 for Scene Class")

    args = parser.parse_args()

    # Pass the argument to your function
    train_stg(args.stage)