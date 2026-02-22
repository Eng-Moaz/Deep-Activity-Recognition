"""Extract ResNet-50 features for all player crops.

Uses the trained B3 Stage 1 backbone (frozen), iterates over all
scenecrops_temporal samples, and saves (features, label) tensors per split.

Output shape per sample: (9, 12, 2048) — 9 frames × 12 players × 2048-dim.

Usage:
    python scripts/extract_features.py [--checkpoint PATH] [--output_dir PATH]
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
from modeling.b3.model import Baseline3_stg1


def build_feature_extractor(checkpoint_path, device):
    """Load B3 Stage 1 and strip FC to get ResNet-50 feature extractor."""
    # Reconstruct B3 Stage 1
    cfg = type("Cfg", (), {"num_classes": 9, "dropout": 0.5})()
    model = Baseline3_stg1(cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Strip FC head → pure ResNet backbone outputting (B, 2048, 1, 1)
    backbone = nn.Sequential(*list(model.backbone.children())[:-1])
    backbone.eval()
    backbone.to(device)
    return backbone


def extract_split(backbone, split, videos_dir, tracks_dir, device, batch_size_images=256):
    """Extract features for one split. Returns list of (features, label)."""

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

    # Process one sample at a time (images are huge)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_features = []
    print(f"\n[{split.upper()}] Extracting features from {len(dataset)} samples...")

    for imgs, label in tqdm(loader, desc=f"  {split}"):
        # imgs: (1, 9, 12, 3, 224, 224)
        imgs = imgs.squeeze(0)  # (9, 12, 3, 224, 224)
        seq_len, n_players, c, h, w = imgs.shape

        flat = imgs.view(seq_len * n_players, c, h, w)  # (108, 3, 224, 224)

        # Process in chunks to avoid OOM
        feat_chunks = []
        for i in range(0, flat.shape[0], batch_size_images):
            chunk = flat[i : i + batch_size_images].to(device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                feat = backbone(chunk)  # (chunk_size, 2048, 1, 1)
            feat_chunks.append(feat.flatten(1).cpu())  # (chunk_size, 2048)

        features = torch.cat(feat_chunks, dim=0)  # (108, 2048)
        features = features.view(seq_len, n_players, 2048)  # (9, 12, 2048)

        all_features.append((features, label.item()))

    return all_features


def main():
    parser = argparse.ArgumentParser(description="Extract ResNet-50 features")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/b3/best_model_b3_stg1.pth",
        help="Path to B3 Stage 1 checkpoint",
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
    print(f"Loading backbone from: {args.checkpoint}")

    backbone = build_feature_extractor(args.checkpoint, device)

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        features = extract_split(
            backbone, split, args.videos_dir, args.tracks_dir, device
        )
        save_path = os.path.join(args.output_dir, f"{split}_features.pt")
        torch.save(features, save_path)
        print(f"  Saved {len(features)} samples → {save_path}")

    print("\nDone! Features saved to:", args.output_dir)


if __name__ == "__main__":
    main()
