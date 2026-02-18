"""Dataloader factory — routes to correct dataset based on explicit config fields."""

from torch.utils.data import DataLoader
from torchvision import transforms

from data_utils.dataset import VolleyballPlayerDataset, VolleyballSceneDataset


def get_data_loaders(cfg):
    """Build train/val/test DataLoaders from config."""

    # Training transform — with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.05),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Val/Test transform — clean
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract explicit routing fields
    task = cfg.task_type
    mode = cfg.input_mode
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    videos_dir = cfg.videos_dir
    tracks_dir = cfg.tracks_dir

    print(f"Loading Data for: {cfg.experiment_name} | Task: {task} | Mode: {mode}")

    # Instantiate Datasets
    if task == "scene":
        train_ds = VolleyballSceneDataset(videos_dir, tracks_dir, "train", mode=mode, transform=transform)
        val_ds = VolleyballSceneDataset(videos_dir, tracks_dir, "val", mode=mode, transform=transform)
        test_ds = VolleyballSceneDataset(videos_dir, tracks_dir, "test", mode=mode, transform=transform)

    elif task == "player":
        train_ds = VolleyballPlayerDataset(videos_dir, tracks_dir, "train", mode=mode, transform=transform)
        val_ds = VolleyballPlayerDataset(videos_dir, tracks_dir, "val", mode=mode, transform=transform)
        test_ds = VolleyballPlayerDataset(videos_dir, tracks_dir, "test", mode=mode, transform=transform)

    else:
        raise ValueError(f"Unknown task_type: {task!r}. Expected 'scene' or 'player'.")

    # Create Loaders
    pin = num_workers > 0
    persist = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persist,
    )

    return train_loader, val_loader, test_loader