# Deep Activity Recognition

Group Activity Recognition on the Volleyball Dataset using deep learning baselines.

---

## Project Structure

```
Deep-Activity-Recognition/
├── modeling/
│   ├── b1/                 # Baseline 1: Spatial scene (ResNet-50)
│   ├── b3/                 # Baseline 3: Two-stage spatial (player → scene)
│   ├── b4/                 # Baseline 4: Temporal scene (ResNet-50 + LSTM)
│   ├── b5/                 # Baseline 5: Temporal two-stage (player LSTM → pool → scene)
│   ├── b6/                 # Baseline 6: Pool players → LSTM temporal (scene)
│   └── b7/                 # Baseline 7: LSTM per player → Pool (pre-extracted features)
├── scripts/
│   └── extract_features.py # One-time ResNet feature extraction for B7
├── data_utils/
│   ├── dataset.py          # All datasets: Scene, Player, and Features
│   └── dataloader.py       # Factory: routes config → correct dataset
├── training/
│   ├── engine.py           # Shared train/validate/evaluate loops
│   └── reproducibility.py  # Seeds, metadata dumps
├── helper_utils/
│   └── evaluation.py       # Confusion matrix, classification report
├── features/               # Pre-extracted features (created by extract_features.py)
├── checkpoints/            # Saved checkpoints (created at runtime)
└── pyproject.toml
```

---

## Baselines Overview

| Baseline | Architecture | Input Mode | Data Shape | Task |
|----------|-------------|------------|------------|------|
| **B1** | ResNet-50 → FC | `scenefull` | `(B, 3, 224, 224)` | Scene |
| **B3 Stg1** | ResNet-50 → FC | `action_train` | `(B, 3, 224, 224)` | Player |
| **B3 Stg2** | Pool players → Pool time → FC | `features` | `(B, 9, 12, 2048)` | Scene |
| **B4** | LSTM frames → FC | `features` | `(B, 9, 2048)` | Scene |
| **B5 Stg1** | ResNet → LSTM per player → FC | `temporal` | `(B, 9, 3, 224, 224)` | Player |
| **B5 Stg2** | LSTM per player → Concat → Pool → FC | `features` | `(B, 9, 12, 2048)` | Scene |
| **B6** | Pool players → LSTM → FC | `features` | `(B, 9, 12, 2048)` | Scene |
| **B7** | LSTM per player → Pool → FC | `features` | `(B, 9, 12, 2048)` | Scene |

---

## Running on Kaggle

### 1. Setup

Upload or clone the repo, then attach the volleyball dataset:

```python
import os
os.chdir("/kaggle/working/Deep-Activity-Recognition")
```

### 2. Training Commands

```bash
# Baseline 1 — Spatial scene classification
python -m modeling.b1.train

# Baseline 3 — Two-stage (run Stage 1 first)
python -m modeling.b3.train --stage 1    # player action classification
python -m modeling.b3.train --stage 2    # scene on features (after extraction)

# Baseline 5 — Temporal two-stage (run Stage 1 first)
python -m modeling.b5.train --stage 1    # player temporal LSTM
python -m modeling.b5.train --stage 2    # group activity on features (after extraction)

# ---- Feature-based baselines (B3 Stg2, B4, B5 Stg2, B6, B7) ----
# Step 1: Extract features (one-time per mode)
python scripts/extract_features.py --mode frame --checkpoint checkpoints/b1/best_model.pth     # for B4
python scripts/extract_features.py --mode player --checkpoint checkpoints/b3/best_model_b3_stg1.pth  # for B3 Stg2, B5 Stg2, B6, B7

# Step 2: Train (fast, batch_size=64)
python -m modeling.b4.train    # frame features
python -m modeling.b6.train    # player features
python -m modeling.b7.train    # player features
```

### Training Order

Some baselines depend on checkpoints from earlier ones:

```
B1 ──► Feature extraction (--mode frame) ──► B4
B3 Stage 1 ──► Feature extraction (--mode player) ──► B3 Stage 2
                                             ├──► B5 Stage 2
                                             ├──► B6
                                             └──► B7
B5 Stage 1 (standalone, no dependencies)
```

### 3. Path Configuration

By default, paths point to standard Kaggle dataset locations. Override via environment variables:

```python
os.environ["VOLLEYBALL_VIDEOS_DIR"] = "/kaggle/input/volleyball/volleyball_/videos"
os.environ["VOLLEYBALL_TRACKS_DIR"] = "/kaggle/input/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation"
os.environ["VOLLEYBALL_FEATURES_DIR"] = "features"  # for B4, B6, B7
```

Or edit `videos_dir` / `tracks_dir` directly in each baseline's `config.py`.

---

## Dataset Modes

| Mode | Returns | Used By |
|------|---------|---------|
| `scenefull` | `(3, 224, 224)` — single full frame | B1 |
| `action_train` | `(3, 224, 224)` — single player crop | B3 Stage 1 |
| `scenecrops` | `(12, 3, 224, 224)` — 12 player crops | B3 Stage 2 |
| `scenefull_temporal` | `(9, 3, 224, 224)` — 9-frame sequence | extract_features --mode frame |
| `temporal` | `(9, 3, 224, 224)` — 9-frame player sequence | B5 Stage 1 |
| `scenecrops_temporal` | `(9, 12, 3, 224, 224)` — 9 frames × 12 crops | B5 Stage 2, extract_features --mode player |
| `features` | `(9, 2048)` or `(9, 12, 2048)` — pre-extracted | B4, B6, B7 |

---

## Feature Extraction Workflow (B3 Stg2, B4, B5 Stg2, B6, B7)

B3 Stg2, B4, B5 Stg2, B6, and B7 use **pre-extracted ResNet-50 features** instead of running the CNN in the training loop.
This enables `batch_size=64` and each model trains in minutes.

### Extract Features (one-time)

```bash
# Frame-level features for B4 — uses B1 backbone
python scripts/extract_features.py --mode frame --checkpoint checkpoints/b1/best_model.pth

# Player-level features for B3 Stg2, B5 Stg2, B6, B7 — uses B3 Stage 1 backbone
python scripts/extract_features.py --mode player --checkpoint checkpoints/b3/best_model_b3_stg1.pth
```

This creates:
```
features/
├── train_frame_features.pt     # (9, 2048)         — for B4
├── val_frame_features.pt
├── test_frame_features.pt
├── train_features.pt           # (9, 12, 2048)     — for B3 Stg2, B5 Stg2, B6, B7
├── val_features.pt
└── test_features.pt
```

### Train

```bash
python -m modeling.b3.train --stage 2   # Pool players → Pool time → FC
python -m modeling.b4.train             # LSTM on frame features
python -m modeling.b5.train --stage 2   # LSTM per player → Concat → Pool → FC
python -m modeling.b6.train             # Pool players → LSTM
python -m modeling.b7.train             # LSTM per player → Pool
```

---

## Config Reference

Each baseline has a `config.py` dataclass. All models take `cfg` as their only constructor argument. Key fields:

| Field | Description |
|-------|-------------|
| `task_type` | `"scene"`, `"player"`, or `"features"` |
| `input_mode` | Dataset mode (see table above) |
| `num_classes` | 8 (scene) or 9 (player actions) |
| `hidden_size` | LSTM hidden dimension (B4, B5, B6, B7) |
| `lstm_layers` | Number of LSTM layers (B4, B5, B6, B7) |
| `seed` | Random seed for reproducibility |
| `model_save_path` | Where to save the best checkpoint |

---

## Evaluation

Evaluation runs **automatically** at the end of every training run via `training/engine.py`. After the training loop, the engine:

1. Reloads the best checkpoint
2. Runs `evaluate_test_set()` from `helper_utils/evaluation.py`
3. Prints classification report + saves confusion matrix PNG

---

## Reproducibility

Every training run automatically:
- Sets seeds (`torch`, `numpy`, `random`, CUDA)
- Enables deterministic mode
- Dumps metadata JSON (config, git SHA, timestamp) to `checkpoints/`
