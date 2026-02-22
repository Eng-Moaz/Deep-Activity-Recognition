import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config_stg1:
    # Experiment
    experiment_name: str = "baseline3_stage1"

    # System
    device: str = "cuda"
    num_workers: int = 8

    # Data
    data_root: str = "/kaggle/input/volleyball/volleyball_"
    task_type: str = "player"
    input_mode: str = "action_train"

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
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    optimizer: str = "AdamW"
    use_amp: bool = True
    patience: int = 5

    # Model
    num_classes: int = 9
    dropout: float = 0.5
    class_names: List[str] = field(default_factory=lambda: [
        'blocking', 'digging', 'falling', 'jumping',
        'moving', 'setting', 'spiking', 'standing', 'waiting',
    ])

    # Scheduler
    use_scheduler: bool = True
    step_size: int = 10
    gamma: float = 0.1

    # Reproducibility
    seed: int = 42

    # Paths
    model_save_path: str = "checkpoints/b3/best_model_b3_stg1.pth"
    cm_save_path: str = "checkpoints/b3/confusion_matrix_b3_stg1.png"


@dataclass
class Config_stg2:
    # Experiment
    experiment_name: str = "baseline3_stage2"

    # System
    device: str = "cuda"
    num_workers: int = 4

    # Data — uses pre-extracted player features: (9 frames x 12 players x 2048)
    task_type: str = "features"
    input_mode: str = "features"
    input_size: int = 2048

    # Feature paths
    features_dir: str = os.environ.get("VOLLEYBALL_FEATURES_DIR", "features")

    # Training
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    use_amp: bool = False
    patience: int = 7
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.0

    # Model
    num_classes: int = 8
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
    scheduler_patience: int = 3
    gamma: float = 0.1

    # Reproducibility
    seed: int = 42

    # Paths
    model_save_path: str = "checkpoints/b3/best_model_b3_stg2.pth"
    cm_save_path: str = "checkpoints/b3/confusion_matrix_b3_stg2.png"