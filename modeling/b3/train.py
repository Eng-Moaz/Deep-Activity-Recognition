"""Baseline 3 — Two-stage: player action recognition → scene classification."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modeling.b3.config import Config_stg1, Config_stg2
from modeling.b3.model import Baseline3_stg1, Baseline3_stg2
from data_utils.dataloader import get_data_loaders
from data_utils.dataset import FeaturesDataset
from torch.utils.data import DataLoader
from training.engine import run_training


def get_feature_loaders(cfg):
    """Build DataLoaders from pre-extracted player feature .pt files."""
    features_dir = cfg.features_dir
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    print(f"Loading player features from: {features_dir}")

    train_ds = FeaturesDataset(os.path.join(features_dir, "train_features.pt"))
    val_ds = FeaturesDataset(os.path.join(features_dir, "val_features.pt"))
    test_ds = FeaturesDataset(os.path.join(features_dir, "test_features.pt"))

    pin = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader


def train_stg(stage):
    if stage == 1:
        cfg = Config_stg1()
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, test_loader = get_data_loaders(cfg)
        model = Baseline3_stg1(cfg).to(device)
    else:
        cfg = Config_stg2()
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, test_loader = get_feature_loaders(cfg)
        model = Baseline3_stg2(cfg).to(device)

    print(f"Device: {device}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    run_training(cfg, model, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline 3 Stages")
    parser.add_argument(
        "--stage", type=int, default=1, choices=[1, 2],
        help="Which stage to train: 1 for Player Actions, 2 for Scene Class",
    )
    args = parser.parse_args()
    train_stg(args.stage)