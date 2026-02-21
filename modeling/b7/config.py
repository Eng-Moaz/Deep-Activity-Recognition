import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Experiment
    experiment_name: str = "baseline7"

    # System
    device: str = "cuda"
    num_workers: int = 8

    # Data — uses scenecrops_temporal: (9 frames x 12 player crops)
    task_type: str = "scene"
    input_mode: str = "scenecrops_temporal"

    # Paths
    videos_dir: str = os.environ.get(
        "VOLLEYBALL_VIDEOS_DIR",
        "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    )
    tracks_dir: str = os.environ.get(
        "VOLLEYBALL_TRACKS_DIR",
        "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    )

    # Training
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    optimizer: str = "AdamW"
    use_amp: bool = True
    patience: int = 7
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.0

    # Model
    num_classes: int = 8
    num_classes_stg1: int = 9   # needed to reconstruct B5 Stage 1 for weight loading
    hidden_size: int = 512      # LSTM_2 hidden dimension
    lstm_layers: int = 1        # LSTM_2 layers
    hidden_size_stg1: int = 128 # must match B5 Stage 1's hidden_size
    lstm_layers_stg1: int = 1   # must match B5 Stage 1's lstm_layers
    dropout: float = 0.5

    class_names: List[str] = field(default_factory=lambda: [
        'l_pass', 'r_pass',
        'l_spike', 'r_spike',
        'l_set', 'r_set',
        'l_winpoint', 'r_winpoint',
    ])

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "ReduceLROnPlateau"
    scheduler_patience: int = 2
    gamma: float = 0.1

    # Reproducibility
    seed: int = 42

    # Paths
    saved_stg1_path: str = "checkpoints/b5/best_model_b5_stg1.pth"
    model_save_path: str = "checkpoints/b7/best_model_b7.pth"
    cm_save_path: str = "checkpoints/b7/confusion_matrix_b7.png"
