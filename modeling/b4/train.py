"""Baseline 4 — Temporal scene classification (ResNet-50 + LSTM on frame sequences)."""

import torch

from modeling.b4.config import Config
from modeling.b4.model import Baseline4
from data_utils.dataloader import get_data_loaders
from training.engine import run_training


def train():
    cfg = Config()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {cfg.experiment_name}")

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # Model
    print(f"Loading Backbone from: {cfg.trained_resnet_path}")
    model = Baseline4(cfg).to(device)

    # Train (shared engine handles optimizer, scheduler, checkpointing, evaluation)
    run_training(cfg, model, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    train()