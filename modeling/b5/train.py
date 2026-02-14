import argparse

import torch

from modeling.b5.config import Config_stg1, Config_stg2
from modeling.b5.model import Baseline5_stg1, Baseline5_stg2
from data_utils.dataloader import get_data_loaders
from training.engine import run_training


def train_stg(stage):
    cfg = Config_stg1() if stage == 1 else Config_stg2()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {cfg.experiment_name}")

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # Model
    if stage == 1:
        model = Baseline5_stg1(cfg).to(device)
    else:
        print(f"Loading Stage 1 backbone from: {cfg.saved_stg1_path}")
        model = Baseline5_stg2(cfg).to(device)

    # Train (shared engine handles optimizer, scheduler, checkpointing, evaluation)
    run_training(cfg, model, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline 5 Stages")
    parser.add_argument(
        "--stage", type=int, default=1, choices=[1, 2],
        help="Which stage to train: 1 for Person Temporal, 2 for Group Activity",
    )
    args = parser.parse_args()
    train_stg(args.stage)
