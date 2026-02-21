import torch

from modeling.b7.config import Config
from modeling.b7.model import Baseline7
from data_utils.dataloader import get_data_loaders
from training.engine import run_training


def main():
    cfg = Config()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {cfg.experiment_name}")

    # Data
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    print(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # Model
    print(f"Loading frozen B5 Stage 1 from: {cfg.saved_stg1_path}")
    model = Baseline7(cfg).to(device)

    # Train
    run_training(cfg, model, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    main()
