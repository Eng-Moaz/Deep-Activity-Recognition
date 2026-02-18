import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Experiment
    experiment_name: str = "baseline4_temporal"

    # System
    device: str = "cuda"
    num_workers: int = 4

    # Data
    data_root: str = "/kaggle/input/volleyball/volleyball_"
    task_type: str = "scene"
    input_mode: str = "scenefull_temporal"     # Sequence of 9 frames

    # Paths — portable via env vars, Kaggle defaults
    videos_dir: str = os.environ.get(
        "VOLLEYBALL_VIDEOS_DIR",
        "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    )
    tracks_dir: str = os.environ.get(
        "VOLLEYBALL_TRACKS_DIR",
        "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    )

    # Training
    epochs: int = 35
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    optimizer: str = "AdamW"
    use_amp: bool = True
    patience: int = 10
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.15

    # Model (LSTM Specifics)
    num_classes: int = 8
    hidden_size: int = 512
    lstm_layers: int = 1
    dropout: float = 0.5

    class_names: List[str] = field(default_factory=lambda: [
        'l_pass', 'r_pass',
        'l_spike', 'r_spike',
        'l_set', 'r_set',
        'l_winpoint', 'r_winpoint'
    ])

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "ReduceLROnPlateau"
    scheduler_patience: int = 2
    gamma: float = 0.1

    # Reproducibility
    seed: int = 42

    # Paths
    trained_resnet_path: str = "checkpoints/b1/best_model.pth"
    model_save_path: str = "checkpoints/b4/best_model_b4.pth"
    cm_save_path: str = "checkpoints/b4/confusion_matrix_b4.png"
