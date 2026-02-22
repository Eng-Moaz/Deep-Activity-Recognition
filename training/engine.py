"""Shared training engine"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from helper_utils.evaluation import evaluate_test_set
from training.reproducibility import dump_run_metadata, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip_norm=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    loop = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward (with AMP)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward (with AMP)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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


def validate(model, loader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
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
    scheduler_type = getattr(cfg, "scheduler_type", "StepLR")
    if scheduler_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(cfg, "gamma", 0.1),
            patience=getattr(cfg, "scheduler_patience", 2),
        )
    elif scheduler_type == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=getattr(cfg, "epochs", 50),
            eta_min=getattr(cfg, "min_lr", 1e-6),
        )
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
    label_smoothing = getattr(cfg, "label_smoothing", 0.0)
    class_weights = getattr(cfg, "class_weights", None)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"[INFO] Class Weights = {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"[INFO] Label Smoothing = {label_smoothing}")

    # Gradient Clipping
    grad_clip_norm = getattr(cfg, "grad_clip_norm", 0.0)
    if grad_clip_norm > 0:
        print(f"[INFO] Gradient Clipping max_norm = {grad_clip_norm}")

    # Mixed Precision
    use_amp = getattr(cfg, "use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("[INFO] Mixed Precision (AMP) enabled")

    # Early Stopping
    patience = getattr(cfg, "patience", 0)
    epochs_no_improve = 0

    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{cfg.experiment_name}")

    # Training Loop
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip_norm=grad_clip_norm,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, use_amp=use_amp,
        )

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
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early Stopping
        if patience > 0 and epochs_no_improve >= patience:
            print(f"    Early stopping triggered (no improvement for {patience} epochs)")
            break

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
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
