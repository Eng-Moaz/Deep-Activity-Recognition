import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config_stg1:
    # Experiment
    experiment_name: str = "baseline5_stage1"

    # System
    device: str = "cuda"
    num_workers: int = 4

    # Data — uses player temporal sequences (9 frames per player)
    task_type: str = "player"
    input_mode: str = "temporal"

    # Paths
    videos_dir: str = os.environ.get(
        "VOLLEYBALL_VIDEOS_DIR",
        "/kaggle/input/volleyball/volleyball_/videos",
    )
    tracks_dir: str = os.environ.get(
        "VOLLEYBALL_TRACKS_DIR",
        "/kaggle/input/volleyball/volleyball_tracking_annotation/volleyball_tracking_annotation",
    )

    # Training
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    optimizer: str = "AdamW"

    # Model
    num_classes: int = 9
    hidden_size: int = 256
    lstm_layers: int = 1
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
    model_save_path: str = "checkpoints/b5/best_model_b5_stg1.pth"
    cm_save_path: str = "checkpoints/b5/confusion_matrix_b5_stg1.png"


@dataclass
class Config_stg2(Config_stg1):
    # Experiment
    experiment_name: str = "baseline5_stage2"

    # Data — uses scenecrops_temporal: (9 frames x 12 player crops)
    task_type: str = "scene"
    input_mode: str = "scenecrops_temporal"

    # Training
    epochs: int = 12
    batch_size: int = 4    # smaller batch — (B, 9, 12, 3, 224, 224) is large
    learning_rate: float = 5e-4

    # Model
    num_classes: int = 8
    num_classes_stg1: int = 9  # needed to load Stage 1 checkpoint
    class_names: List[str] = field(default_factory=lambda: [
        'l_pass', 'r_pass',
        'l_spike', 'r_spike',
        'l_set', 'r_set',
        'l_winpoint', 'r_winpoint',
    ])

    # Paths
    saved_stg1_path: str = "checkpoints/b5/best_model_b5_stg1.pth"
    model_save_path: str = "checkpoints/b5/best_model_b5_stg2.pth"
    cm_save_path: str = "checkpoints/b5/confusion_matrix_b5_stg2.png"
