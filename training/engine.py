"""Shared training engine"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from helper_utils.evaluation import evaluate_test_set
from training.reproducibility import dump_run_metadata, set_seed


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

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
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
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def _build_optimizer(cfg, parameters):
    optimizer_name = getattr(cfg, "optimizer", "AdamW")
    if optimizer_name == "AdamW":
        return optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif optimizer_name == "SGD":
        momentum = getattr(cfg, "momentum", 0.9)
        return optim.SGD(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _build_scheduler(cfg, optimizer):
    if not getattr(cfg, "use_scheduler", False):
        return None
    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.step_size,
        gamma=cfg.gamma,
    )


def run_training(cfg, model, train_loader, val_loader, test_loader, device):
    """Shared training loop: train → validate → checkpoint → evaluate."""

    # Reproducibility
    seed = getattr(cfg, "seed", 42)
    set_seed(seed)

    # Dump run metadata next to the saved model
    save_dir = os.path.dirname(cfg.model_save_path)
    os.makedirs(save_dir, exist_ok=True)
    dump_run_metadata(cfg, save_dir)

    # Optimizer + Scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = _build_optimizer(cfg, trainable_params)
    scheduler = _build_scheduler(cfg, optimizer)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{cfg.experiment_name}")

    # Training Loop
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"    Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"    Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # TensorBoard logging
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"    Saved Best Model ({best_acc:.2f}%)")

        if scheduler is not None:
            scheduler.step()

    # Final Evaluation
    print("\nEvaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))

    test_acc = evaluate_test_set(
        model,
        test_loader,
        device,
        cfg.class_names,
        cfg.experiment_name,
        cfg.cm_save_path,
    )
    print(f"FINAL TEST ACCURACY: {test_acc * 100:.2f}%")

    writer.close()
    return test_acc
