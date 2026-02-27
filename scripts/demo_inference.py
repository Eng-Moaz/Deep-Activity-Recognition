"""Demo: Group activity recognition video with player tracking and predictions.

Plays through ALL frames in a clip at natural speed, runs the B8 model
on each 9-frame window, and overlays player boxes + group activity predictions.

Usage:
    python scripts/demo_inference.py \
        --video_id 4 --clip_id 29211 \
        --backbone_ckpt checkpoints/b3/best_model_b3_stg1.pth \
        --model_ckpt checkpoints/b8/best_model_b8.pth \
        --output demo.mp4
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CLASS_NAMES = [
    "Left Pass", "Right Pass", "Left Spike", "Right Spike",
    "Left Set", "Right Set", "Left Winpoint", "Right Winpoint",
]

ACTION_CLASSES = [
    "blocking", "digging", "falling", "jumping",
    "moving", "setting", "spiking", "standing", "waiting",
]

MAX_PLAYERS = 12

# Colors (BGR)
BOX_COLOR = (136, 255, 0)
ACTIVITY_BG = (30, 30, 30)
ACTIVITY_COLOR = (255, 200, 0)
INFO_COLOR = (200, 200, 200)


def load_backbone(checkpoint_path, device):
    from modeling.b3.model import Baseline3_stg1
    cfg = type("Cfg", (), {"num_classes": 9, "dropout": 0.5})()
    model = Baseline3_stg1(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    backbone = nn.Sequential(*list(model.backbone.children())[:-1])
    backbone.eval().to(device)
    return backbone


def load_action_model(checkpoint_path, device):
    from modeling.b3.model import Baseline3_stg1
    cfg = type("Cfg", (), {"num_classes": 9, "dropout": 0.5})()
    model = Baseline3_stg1(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval().to(device)
    return model


def load_model(model_name, checkpoint_path, device):
    if model_name == "b8":
        from modeling.b8.config import Config
        from modeling.b8.model import Baseline8
        cfg = Config()
        model = Baseline8(cfg)
    elif model_name == "b7":
        from modeling.b7.config import Config
        from modeling.b7.model import Baseline7
        cfg = Config()
        model = Baseline7(cfg)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval().to(device)
    return model


def load_all_frames(videos_dir, tracks_dir, video_id, clip_id):
    """Load ALL frames in the clip directory with their tracking boxes."""
    clip_dir = os.path.join(videos_dir, str(video_id), str(clip_id))
    track_file = os.path.join(tracks_dir, str(video_id), str(clip_id), f"{clip_id}.txt")

    # Parse tracking
    frames_data = defaultdict(list)
    if os.path.exists(track_file):
        with open(track_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                try:
                    fid = int(parts[5])
                    lost = int(parts[6])
                    if lost == 1:
                        continue
                    box = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                    frames_data[fid].append(box)
                except (ValueError, IndexError):
                    continue

    # Get all frame images, sorted by frame ID
    frame_files = sorted(
        [f for f in os.listdir(clip_dir) if f.endswith(".jpg")],
        key=lambda x: int(x.split(".")[0]),
    )

    frames = []
    for fname in frame_files:
        fid = int(fname.split(".")[0])
        frames.append({
            "path": os.path.join(clip_dir, fname),
            "fid": fid,
            "boxes": frames_data.get(fid, []),
        })

    return frames


def extract_window_features(backbone, frames_window, device):
    """Extract features for a 9-frame window: (1, 9, 12, 2048)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_timesteps = []
    for frame in frames_window:
        img = Image.open(frame["path"]).convert("RGB")
        crops = []
        for box in frame["boxes"]:
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(img.width, box[2]), min(img.height, box[3])
            if x2 > x1 and y2 > y1:
                crops.append(transform(img.crop((x1, y1, x2, y2))))

        crops = crops[:MAX_PLAYERS]
        while len(crops) < MAX_PLAYERS:
            crops.append(torch.zeros(3, 224, 224))

        crop_batch = torch.stack(crops).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            feats = backbone(crop_batch).flatten(1)
        all_timesteps.append(feats.cpu())

    return torch.stack(all_timesteps).unsqueeze(0).float()


def predict_actions_batch(action_model, frame, device):
    """Predict individual player actions for a single frame."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(frame["path"]).convert("RGB")
    crops = []
    valid_indices = []

    for i, box in enumerate(frame["boxes"]):
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(img.width, box[2]), min(img.height, box[3])
        if x2 > x1 and y2 > y1:
            crops.append(transform(img.crop((x1, y1, x2, y2))))
            valid_indices.append(i)

    if not crops:
        return {}

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        logits = action_model(batch)
        preds = torch.argmax(logits, dim=1).cpu().tolist()

    return {valid_indices[i]: ACTION_CLASSES[preds[i]] for i in range(len(valid_indices))}


def draw_frame(img_bgr, boxes, activity, confidence, actions, frame_idx, total_frames, has_prediction):
    """Draw tracking boxes, activity prediction, and player actions on a frame."""
    h, w = img_bgr.shape[:2]

    # Draw player bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Action label above each player
        if actions and i in actions:
            action = actions[i]
            label_size = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            lx, ly = x1, y1 - 5
            cv2.rectangle(img_bgr, (lx, ly - label_size[1] - 4), (lx + label_size[0] + 4, ly + 2), (0, 0, 0), -1)
            cv2.putText(img_bgr, action, (lx + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Activity banner at the top
    banner_h = 50
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), ACTIVITY_BG, -1)
    cv2.addWeighted(overlay, 0.8, img_bgr, 0.2, 0, img_bgr)

    if has_prediction:
        text = f"Activity: {activity}  ({confidence:.1f}%)"
        cv2.putText(img_bgr, text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ACTIVITY_COLOR, 2, cv2.LINE_AA)
    else:
        cv2.putText(img_bgr, "Collecting frames...", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INFO_COLOR, 2, cv2.LINE_AA)

    # Frame counter
    counter = f"Frame {frame_idx + 1}/{total_frames}"
    cv2.putText(img_bgr, counter, (w - 160, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INFO_COLOR, 1, cv2.LINE_AA)

    return img_bgr


def main():
    parser = argparse.ArgumentParser(description="Demo: group activity recognition video")
    parser.add_argument("--video_id", type=int, required=True)
    parser.add_argument("--clip_id", type=int, required=True)
    parser.add_argument("--model", default="b8", choices=["b7", "b8"])
    parser.add_argument("--backbone_ckpt", default="checkpoints/b3/best_model_b3_stg1.pth")
    parser.add_argument("--model_ckpt", default="checkpoints/b8/best_model_b8.pth")
    parser.add_argument("--output", default="demo.mp4")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS")
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
    print(f"Model:  {args.model.upper()}")

    # Load models
    print("Loading backbone...")
    backbone = load_backbone(args.backbone_ckpt, device)
    print("Loading action model...")
    action_model = load_action_model(args.backbone_ckpt, device)
    print("Loading scene model...")
    model = load_model(args.model, args.model_ckpt, device)

    # Load ALL frames in the clip
    print(f"Loading clip: video={args.video_id}, clip={args.clip_id}")
    frames = load_all_frames(args.videos_dir, args.tracks_dir, args.video_id, args.clip_id)
    print(f"  Found {len(frames)} frames")

    # Run predictions on every valid 9-frame window
    print("Running predictions on sliding windows...")
    predictions = {}  # frame_idx -> (activity, confidence)
    window_size = 9

    for start in range(len(frames) - window_size + 1):
        window = frames[start : start + window_size]
        center_idx = start + window_size // 2

        features = extract_window_features(backbone, window, device)
        with torch.no_grad():
            logits = model(features.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        activity = CLASS_NAMES[pred.item()]
        confidence = conf.item() * 100
        predictions[center_idx] = (activity, confidence)
        print(f"  Window {start}-{start + window_size - 1} -> {activity} ({confidence:.1f}%)")

    # Predict player actions per frame
    print("Predicting player actions...")
    all_actions = {}
    for i, frame in enumerate(frames):
        all_actions[i] = predict_actions_batch(action_model, frame, device)

    # Render video
    print(f"Rendering video ({len(frames)} frames @ {args.fps} FPS)...")
    sample_img = cv2.imread(frames[0]["path"])
    h, w = sample_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    # Track the current prediction (carry forward from last known window)
    current_activity = "..."
    current_confidence = 0.0

    for i, frame in enumerate(frames):
        if i in predictions:
            current_activity, current_confidence = predictions[i]

        img_bgr = cv2.imread(frame["path"])
        has_prediction = i >= window_size // 2
        annotated = draw_frame(
            img_bgr, frame["boxes"],
            current_activity, current_confidence,
            all_actions.get(i, {}),
            i, len(frames), has_prediction,
        )
        writer.write(annotated)

    writer.release()
    duration = len(frames) / args.fps
    print(f"\nVideo saved: {args.output}")
    print(f"  {len(frames)} frames, {duration:.1f}s @ {args.fps} FPS")
    print(f"  {len(predictions)} activity predictions across sliding windows")


if __name__ == "__main__":
    main()
