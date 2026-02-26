"""Extract features for downstream temporal baselines.

Three extraction modes:
  - player:          (9, 12, 2048)              — ResNet features per player (B5 stg2, B6, B7)
  - player_temporal: (9, 12, 2048 + lstm_hidden) — ResNet + LSTM concat (B5 stg2, B6, B7, B8)
  - frame:           (9, 2048)                   — ResNet features per frame (B4)

Usage:
    # Player-level ResNet features — uses B3 Stage 1 backbone
    python scripts/extract_features.py --mode player --checkpoint checkpoints/b3/best_model_b3_stg1.pth

    # Player-level temporal features — uses B5 Stage 1 (backbone + LSTM)
    python scripts/extract_features.py --mode player_temporal --checkpoint checkpoints/b5/best_model_b5_stg1.pth

    # Frame-level features — uses B1 backbone
    python scripts/extract_features.py --mode frame --checkpoint checkpoints/b1/best_model.pth

    # Player-level sequence features (spatial only) — uses B3 Stage 1 backbone (For Baseline 6)
    python scripts/extract_features.py --mode spatial_sequence --checkpoint checkpoints/b3/best_model_b3_stg1.pth
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

    # If the backbone is already stripped (Baseline 5 stg1), use it directly.
    # Otherwise, strip the FC head.
    if isinstance(model.backbone, nn.Sequential):
        backbone = model.backbone
    else:
        backbone = nn.Sequential(*list(model.backbone.children())[:-1])
    
    backbone.eval()
    backbone.to(device)
    return backbone


def build_temporal_extractor(checkpoint_path, device):
    """Load a trained B5 Stage 1 model (backbone + LSTM) for temporal feature extraction."""
    from modeling.b5.model import Baseline5_stg1
    from modeling.b5.config import Config_stg1

    cfg = Config_stg1()
    model = Baseline5_stg1(cfg)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    backbone = model.backbone
    backbone.eval()
    backbone.to(device)

    lstm = model.lstm
    lstm.eval()

    return backbone, lstm, cfg.hidden_size


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


def extract_temporal_features(backbone, lstm, split, videos_dir, tracks_dir, device, batch_size_images=256):
    """Extract ResNet + LSTM features: (9, 12, 2048 + lstm_hidden) per sample."""

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
    print(f"\n[{split.upper()}] Extracting temporal features from {len(dataset)} samples...")

    for imgs, label in tqdm(loader, desc=f"  {split}"):
        # imgs: (1, 9, 12, 3, 224, 224)
        imgs = imgs.squeeze(0)  # (9, 12, 3, 224, 224)
        seq_len, n_players, c, h, w = imgs.shape

        # Extract ResNet features for all players across all frames
        flat = imgs.view(seq_len * n_players, c, h, w)  # (108, 3, 224, 224)

        feat_chunks = []
        for i in range(0, flat.shape[0], batch_size_images):
            chunk = flat[i : i + batch_size_images].to(device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                feat = backbone(chunk)
            feat_chunks.append(feat.flatten(1).cpu())

        cnn_features = torch.cat(feat_chunks, dim=0).float()  # (108, 2048)

        # Reshape to (9, 12, 2048) - frames x players x features
        cnn_reshaped = cnn_features.view(seq_len, n_players, -1)  # (9, 12, 2048)
        
        # For LSTM: need (batch=players, seq=frames, features) for batch_first=True
        # Process all players together - each row is one player's 9-frame sequence
        cnn_for_lstm = cnn_reshaped.permute(1, 0, 2)  # (12, 9, 2048)

        with torch.no_grad():
            lstm_out, _ = lstm(cnn_for_lstm.to(device))  # (12, 9, hidden)
        lstm_out = lstm_out.cpu()

        # According to paper equation 7: P_tk = x_tk ⊕ h_tk
        # Concatenate CNN feature with LSTM hidden state at each timestep
        # Both tensors are (12, 9, ...), concatenate on feature dimension
        combined = torch.cat([cnn_for_lstm, lstm_out], dim=2)  # (12, 9, 2048 + hidden)
        
        combined = combined.permute(1, 0, 2)  # (9, 12, 2048 + hidden)

        all_features.append((combined, label.item()))

    return all_features


def extract_spatial_sequence_features(backbone, split, videos_dir, tracks_dir, device, batch_size_images=256):
    """Extract sequence of CNN features: (9, 12, 2048) per sample without any RNN attached."""

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
    print(f"\n[{split.upper()}] Extracting spatial sequence features from {len(dataset)} samples...")

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
    parser = argparse.ArgumentParser(description="Extract features for temporal baselines")
    parser.add_argument(
        "--mode",
        choices=["player", "player_temporal", "frame", "spatial_sequence"],
        default="player",
        help="'player' (9,12,2048); 'player_temporal' (9,12,2048+H); 'frame' (9,2048); 'spatial_sequence' (9, 12, 2048)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/b3/best_model_b3_stg1.pth",
        help="B3 stg1 for player, B5 stg1 for player_temporal, B1 for frame",
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
    print(f"Loading model from: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "player_temporal":
        backbone, lstm, hidden_size = build_temporal_extractor(args.checkpoint, device)
        feat_dim = 2048 + hidden_size
        print(f"Temporal feature dim: {feat_dim} (2048 CNN + {hidden_size} LSTM)")

        for split in ["train", "val", "test"]:
            features = extract_temporal_features(
                backbone, lstm, split, args.videos_dir, args.tracks_dir, device
            )
            filename = f"{split}_temporal_features.pt"
            save_path = os.path.join(args.output_dir, filename)
            torch.save(features, save_path)
            print(f"  Saved {len(features)} samples -> {save_path}")
    else:
        model_type = "b1" if args.mode == "frame" else "b3"
        backbone = build_feature_extractor(args.checkpoint, model_type, device)

        for split in ["train", "val", "test"]:
            if args.mode == "player":
                features = extract_player_features(
                    backbone, split, args.videos_dir, args.tracks_dir, device
                )
                filename = f"{split}_features.pt"
            elif args.mode == "spatial_sequence":
                features = extract_spatial_sequence_features(
                    backbone, split, args.videos_dir, args.tracks_dir, device
                )
                filename = f"{split}_spatial_sequence_features.pt"
            else:
                features = extract_frame_features(
                    backbone, split, args.videos_dir, args.tracks_dir, device
                )
                filename = f"{split}_frame_features.pt"

            save_path = os.path.join(args.output_dir, filename)
            torch.save(features, save_path)
            print(f"  Saved {len(features)} samples -> {save_path}")

    print(f"\nDone! Features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
