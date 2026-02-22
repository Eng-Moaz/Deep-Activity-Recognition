"""Train Baseline 7 on pre-extracted features."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from modeling.b7.config import Config
from modeling.b7.model import Baseline7
from training.engine import run_training
from data_utils.dataset import FeaturesDataset
from torch.utils.data import DataLoader


def get_feature_loaders(cfg):
    """Build DataLoaders from pre-extracted feature .pt files."""
    features_dir = cfg.features_dir
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    print(f"Loading features from: {features_dir}")

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


if __name__ == "__main__":
    cfg = Config()
    train_loader, val_loader, test_loader = get_feature_loaders(cfg)
    model = Baseline7(cfg)
    run_training(cfg, model, train_loader, val_loader, test_loader)
