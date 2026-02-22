import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Experiment
    experiment_name: str = "baseline4"

    # System
    device: str = "cuda"
    num_workers: int = 4

    # Data — uses pre-extracted frame features: (9 frames x 2048)
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
    hidden_size: int = 1024
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
    scheduler_patience: int = 3
    gamma: float = 0.1

    # Reproducibility
    seed: int = 42

    # Paths
    model_save_path: str = "checkpoints/b4/best_model_b4.pth"
    cm_save_path: str = "checkpoints/b4/confusion_matrix_b4.png"
