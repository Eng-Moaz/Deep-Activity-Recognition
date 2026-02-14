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
│   └── b5/                 # Baseline 5: Temporal two-stage (player LSTM → pool → scene)
├── data_utils/
│   ├── dataset.py          # VolleyballSceneDataset, VolleyballPlayerDataset
│   └── dataloader.py       # Factory: routes config → correct dataset
├── training/
│   ├── engine.py           # Shared train/validate/evaluate loops
│   └── reproducibility.py  # Seeds, metadata dumps
├── helper_utils/
│   └── evaluation.py       # Confusion matrix, classification report
├── checkpoints/            # Saved checkpoints (created at runtime)
└── pyproject.toml
```

---

## Baselines Overview

| Baseline | Architecture | Input Mode | Data Shape | Task |
|----------|-------------|------------|------------|------|
| **B1** | ResNet-50 → FC | `scenefull` | `(B, 3, 224, 224)` | Scene |
| **B3 Stg1** | ResNet-50 → FC | `action_train` | `(B, 3, 224, 224)` | Player |
| **B3 Stg2** | Frozen ResNet → Pool players → FC | `scenecrops` | `(B, 12, 3, 224, 224)` | Scene |
| **B4** | Frozen ResNet → LSTM frames → FC | `scenefull_temporal` | `(B, 9, 3, 224, 224)` | Scene |
| **B5 Stg1** | ResNet → LSTM per player → FC | `temporal` | `(B, 9, 3, 224, 224)` | Player |
| **B5 Stg2** | Frozen ResNet+LSTM per player → Concat → Pool → FC | `scenecrops_temporal` | `(B, 9, 12, 3, 224, 224)` | Scene |

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
python -m modeling.b3.train --stage 2    # scene classification (loads Stage 1)

# Baseline 4 — Temporal scene (requires B1 checkpoint)
python -m modeling.b4.train

# Baseline 5 — Temporal two-stage (run Stage 1 first)
python -m modeling.b5.train --stage 1    # player temporal LSTM
python -m modeling.b5.train --stage 2    # group activity (loads Stage 1)
```

### Training Order

Some baselines depend on checkpoints from earlier ones:

```
B1 ──────────────────────────► B4 (loads B1 checkpoint)
B3 Stage 1 ──► B3 Stage 2
B5 Stage 1 ──► B5 Stage 2
```

### 3. Path Configuration

By default, paths point to standard Kaggle dataset locations. Override via environment variables:

```python
os.environ["VOLLEYBALL_VIDEOS_DIR"] = "/kaggle/input/volleyball/volleyball_/videos"
os.environ["VOLLEYBALL_TRACKS_DIR"] = "/kaggle/input/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation"
```

Or edit `videos_dir` / `tracks_dir` directly in each baseline's `config.py`.

---

## Dataset Modes

| Mode | Returns | Used By |
|------|---------|---------|
| `scenefull` | `(3, 224, 224)` — single full frame | B1 |
| `action_train` | `(3, 224, 224)` — single player crop | B3 Stage 1 |
| `scenecrops` | `(12, 3, 224, 224)` — 12 player crops | B3 Stage 2 |
| `scenefull_temporal` | `(9, 3, 224, 224)` — 9-frame sequence | B4 |
| `temporal` | `(9, 3, 224, 224)` — 9-frame player sequence | B5 Stage 1 |
| `scenecrops_temporal` | `(9, 12, 3, 224, 224)` — 9 frames × 12 crops | B5 Stage 2 |

---

## Config Reference

Each baseline has a `config.py` dataclass. All models take `cfg` as their only constructor argument. Key fields:

| Field | Description |
|-------|-------------|
| `task_type` | `"scene"` or `"player"` |
| `input_mode` | Dataset mode (see table above) |
| `num_classes` | 8 (scene) or 9 (player actions) |
| `hidden_size` | LSTM hidden dimension (B4, B5) |
| `lstm_layers` | Number of LSTM layers (B4, B5) |
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
