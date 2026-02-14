import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Experiment
    experiment_name: str = "baseline1_spatial"

    # System
    device: str = "cuda"
    num_workers: int = 4

    # Data
    data_root: str = "/kaggle/input/volleyball/volleyball_"
    task_type: str = "scene"
    input_mode: str = "scenefull"

    # Paths
    videos_dir: str = os.environ.get(
        "VOLLEYBALL_VIDEOS_DIR",
        "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_/videos",
    )
    tracks_dir: str = os.environ.get(
        "VOLLEYBALL_TRACKS_DIR",
        "/kaggle/input/datasets/ahmedmohamed365/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    )

    # Model
    num_classes: int = 8
    class_names: List[str] = field(default_factory=lambda: [
        'l_pass', 'r_pass',
        'l_spike', 'r_spike',
        'l_set', 'r_set',
        'l_winpoint', 'r_winpoint'
    ])

    # Training
    batch_size: int = 64
    learning_rate: float = 0.001
    dropout: float = 0.6
    optimizer: str = "AdamW"
    weight_decay: float = 0.0001
    epochs: int = 15
    use_amp: bool = True
    patience: int = 5

    # Scheduler
    use_scheduler: bool = False

    # Reproducibility
    seed: int = 42

    # Paths
    model_save_path: str = "checkpoints/b1/best_model.pth"
    cm_save_path: str = "checkpoints/b1/confusion_matrix.png"