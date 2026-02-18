import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Experiment
    experiment_name: str = "baseline6"

    # System
    device: str = "cuda"
    num_workers: int = 4

    # Data uses scenecrops_temporal: (9 frames x 12 player crops)
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
    batch_size: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.001
    optimizer: str = "AdamW"
    use_amp: bool = True
    patience: int = 10

    # Model
    num_classes: int = 8
    num_classes_stg1: int = 9   # needed to reconstruct B3 Stage 1 for weight loading
    hidden_size: int = 256
    lstm_layers: int = 1
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
    saved_resnet50_path: str = "checkpoints/b3/best_model_b3_stg1.pth"
    model_save_path: str = "checkpoints/b6/best_model_b6.pth"
    cm_save_path: str = "checkpoints/b6/confusion_matrix_b6.png"
