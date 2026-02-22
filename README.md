# Deep Activity Recognition

Group activity recognition on the Volleyball Dataset, based on the paper *"Hierarchical Deep Temporal Models for Group Activity Recognition"* (Ibrahim et al., TPAMI 2016).

We implement 8 baselines that progressively build up from a simple image classifier to a full two-stage hierarchical model with team-level pooling.

---

## Baselines

| # | Name | What it does |
|---|------|-------------|
| B1 | Image Classification | ResNet-50 on full frame |
| B3 | Fine-tuned Person Classification | Stage 1: train ResNet on player crops → Stage 2: pool + classify |
| B4 | Temporal Scene Model | LSTM on frame-level features |
| B5 | Temporal Person Model | Stage 1: ResNet + LSTM per player → Stage 2: pool + classify |
| B6 | Scene LSTM (no person LSTM) | Pool players per frame → LSTM over frames |
| B7 | Full Model V1 | Pool all players per frame → scene LSTM → classify |
| B8 | Full Model V2 (team pooling) | Pool each team separately → concat → scene LSTM → classify |

B5 Stage 1 is the backbone for **all temporal baselines** (B5 stg2, B6, B7, B8). It produces temporal features by concatenating ResNet output + LSTM hidden state per player per frame.

---

## Quick Start (Kaggle)

```python
import os
os.chdir("/kaggle/working/Deep-Activity-Recognition")
```

### Training Order

```
1. B1  →  extract frame features (for B4)
2. B3 Stage 1  →  extract player features (for B3 Stage 2)
3. B5 Stage 1  →  extract temporal features (for B5 stg2, B6, B7, B8)
```

### Step-by-step

```bash
# --- Independent models (no dependencies) ---
python -m modeling.b1.train
python -m modeling.b3.train --stage 1
python -m modeling.b5.train --stage 1

# --- Extract features (one-time each) ---
python scripts/extract_features.py --mode frame --checkpoint checkpoints/b1/best_model.pth
python scripts/extract_features.py --mode player --checkpoint checkpoints/b3/best_model_b3_stg1.pth
python scripts/extract_features.py --mode player_temporal --checkpoint checkpoints/b5/best_model_b5_stg1.pth

# --- Feature-based models (fast, trains in minutes) ---
python -m modeling.b3.train --stage 2
python -m modeling.b4.train
python -m modeling.b5.train --stage 2
python -m modeling.b6.train
python -m modeling.b7.train
python -m modeling.b8.train
```

---

## Feature Extraction

Three modes are available in `scripts/extract_features.py`:

| Mode | Output shape | Used by |
|------|-------------|---------|
| `frame` | `(9, 2048)` | B4 |
| `player` | `(9, 12, 2048)` | B3 Stage 2 |
| `player_temporal` | `(9, 12, 3072)` | B5 stg2, B6, B7, B8 |

The `player_temporal` mode loads a frozen B5 Stage 1 and concatenates ResNet features (2048-D) with LSTM hidden states (1024-D) at each timestep — matching the paper's Equation 7.

---

## Project Structure

```
Deep-Activity-Recognition/
├── modeling/
│   ├── b1/          # Image classification
│   ├── b3/          # Two-stage spatial
│   ├── b4/          # Temporal scene
│   ├── b5/          # Temporal person (two-stage)
│   ├── b6/          # Pool → scene LSTM
│   ├── b7/          # Pool all → scene LSTM
│   └── b8/          # Team pool → scene LSTM
├── scripts/
│   └── extract_features.py
├── data_utils/
│   ├── dataset.py
│   └── dataloader.py
├── training/
│   ├── engine.py
│   └── reproducibility.py
├── helper_utils/
│   └── evaluation.py
├── features/        # created by extract_features.py
└── checkpoints/     # saved during training
```

Each baseline folder contains `model.py`, `config.py`, and `train.py`.

---

## Configuration

Override dataset paths via environment variables:

```python
os.environ["VOLLEYBALL_VIDEOS_DIR"] = "/path/to/videos"
os.environ["VOLLEYBALL_TRACKS_DIR"] = "/path/to/tracking_annotations"
os.environ["VOLLEYBALL_FEATURES_DIR"] = "features"
```

Or edit the defaults directly in each baseline's `config.py`.

---

## Evaluation

Evaluation runs automatically after training. The engine reloads the best checkpoint, prints a classification report, and saves a confusion matrix to `checkpoints/`.

---

## Reproducibility

Every run automatically sets seeds for `torch`, `numpy`, `random`, and CUDA, enables deterministic mode, and saves metadata (config, git SHA, timestamp) to `checkpoints/`.
