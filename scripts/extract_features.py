"""Extract ResNet-50 features for downstream temporal baselines.

Two extraction modes:
  - player:  (9, 12, 2048) per sample — for B6, B7 (player-level)
  - frame:   (9, 2048)     per sample — for B4 (frame-level, no players)

Usage:
    # Player-level features (for B6, B7) — uses B3 Stage 1 backbone
    python scripts/extract_features.py --mode player --checkpoint checkpoints/b3/best_model_b3_stg1.pth

    # Frame-level features (for B4) — uses B1 backbone
    python scripts/extract_features.py --mode frame --checkpoint checkpoints/b1/best_model.pth
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_utils.dataset import VolleyballSceneDataset


def build_feature_extractor(checkpoint_path, model_type, device):
    """Load a trained model and strip FC to get ResNet-50 feature extractor."""
    if model_type == "b3":
        from modeling.b3.model import Baseline3_stg1
        cfg = type("Cfg", (), {"num_classes": 9, "dropout": 0.5})()
        model = Baseline3_stg1(cfg)
    elif model_type == "b1":
        from modeling.b1.model import Baseline1
        cfg = type("Cfg", (), {"num_classes": 8, "dropout": 0.5})()
        model = Baseline1(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Strip FC head → pure ResNet backbone outputting (B, 2048, 1, 1)
    backbone = nn.Sequential(*list(model.backbone.children())[:-1])
    backbone.eval()
    backbone.to(device)
    return backbone


def extract_player_features(backbone, split, videos_dir, tracks_dir, device, batch_size_images=256):
    """Extract player-level features: (9, 12, 2048) per sample."""

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VolleyballSceneDataset(
        videos_dir, tracks_dir, split,
        mode="scenecrops_temporal",
        transform=eval_transform,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_features = []
    print(f"\n[{split.upper()}] Extracting player features from {len(dataset)} samples...")

    for imgs, label in tqdm(loader, desc=f"  {split}"):
        # imgs: (1, 9, 12, 3, 224, 224)
        imgs = imgs.squeeze(0)  # (9, 12, 3, 224, 224)
        seq_len, n_players, c, h, w = imgs.shape

        flat = imgs.view(seq_len * n_players, c, h, w)  # (108, 3, 224, 224)

        feat_chunks = []
        for i in range(0, flat.shape[0], batch_size_images):
            chunk = flat[i : i + batch_size_images].to(device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                feat = backbone(chunk)
            feat_chunks.append(feat.flatten(1).cpu())

        features = torch.cat(feat_chunks, dim=0)  # (108, 2048)
        features = features.view(seq_len, n_players, 2048)  # (9, 12, 2048)

        all_features.append((features, label.item()))

    return all_features


def extract_frame_features(backbone, split, videos_dir, tracks_dir, device, batch_size_images=256):
    """Extract frame-level features: (9, 2048) per sample."""

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VolleyballSceneDataset(
        videos_dir, tracks_dir, split,
        mode="scenefull_temporal",
        transform=eval_transform,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_features = []
    print(f"\n[{split.upper()}] Extracting frame features from {len(dataset)} samples...")

    for imgs, label in tqdm(loader, desc=f"  {split}"):
        # imgs: (1, 9, 3, 224, 224)
        imgs = imgs.squeeze(0)  # (9, 3, 224, 224)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            feat = backbone(imgs.to(device))  # (9, 2048, 1, 1)
        features = feat.flatten(1).cpu()  # (9, 2048)

        all_features.append((features, label.item()))

    return all_features


def main():
    parser = argparse.ArgumentParser(description="Extract ResNet-50 features")
    parser.add_argument(
        "--mode",
        choices=["player", "frame"],
        default="player",
        help="'player' → (9,12,2048) for B6/B7; 'frame' → (9,2048) for B4",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/b3/best_model_b3_stg1.pth",
        help="Path to trained checkpoint (B3 Stage 1 for player, B1 for frame)",
    )
    parser.add_argument(
        "--output_dir",
        default="features",
        help="Directory to save extracted features",
    )
    parser.add_argument(
        "--videos_dir",
        default=os.environ.get(
            "VOLLEYBALL_VIDEOS_DIR",
            "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
        ),
    )
    parser.add_argument(
        "--tracks_dir",
        default=os.environ.get(
            "VOLLEYBALL_TRACKS_DIR",
            "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Loading backbone from: {args.checkpoint}")

    # Determine model type from checkpoint path
    model_type = "b1" if args.mode == "frame" else "b3"
    backbone = build_feature_extractor(args.checkpoint, model_type, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Choose prefix based on mode
    prefix = "frame" if args.mode == "frame" else ""

    for split in ["train", "val", "test"]:
        if args.mode == "player":
            features = extract_player_features(
                backbone, split, args.videos_dir, args.tracks_dir, device
            )
            filename = f"{split}_features.pt"
        else:
            features = extract_frame_features(
                backbone, split, args.videos_dir, args.tracks_dir, device
            )
            filename = f"{split}_frame_features.pt"

        save_path = os.path.join(args.output_dir, filename)
        torch.save(features, save_path)
        print(f"  Saved {len(features)} samples → {save_path}")

    print("\nDone! Features saved to:", args.output_dir)


if __name__ == "__main__":
    main()
