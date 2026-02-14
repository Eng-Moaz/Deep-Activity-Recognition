import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config_stg1:
    # Experiment
    experiment_name: str = "baseline3_stage1"

    # System
    device: str = "cuda"
    num_workers: int = 4

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
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.001
    optimizer: str = "AdamW"

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
class Config_stg2(Config_stg1):
    # Experiment
    experiment_name: str = "baseline3_stage2"

    # Data
    task_type: str = "scene"
    input_mode: str = "scenecrops"   # Stage 2: stack of 12 player crops

    # Training
    epochs: int = 12
    learning_rate: float = 5e-4
    batch_size: int = 8

    # Model
    num_classes: int = 8
    num_classes_stg1: int = 9  # needed to load Stage 1 checkpoint
    class_names: List[str] = field(default_factory=lambda: [
        'l_pass', 'r_pass',
        'l_spike', 'r_spike',
        'l_set', 'r_set',
        'l_winpoint', 'r_winpoint'
    ])

    # Paths
    saved_resnet50_path: str = "checkpoints/b3/best_model_b3_stg1.pth"
    model_save_path: str = "checkpoints/b3/best_model_b3_stg2.pth"
    cm_save_path: str = "checkpoints/b3/confusion_matrix_b3_stg2.png"