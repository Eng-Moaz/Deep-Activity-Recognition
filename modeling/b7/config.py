import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Experiment
    experiment_name: str = "baseline7"

    # System
    device: str = "cuda"
    num_workers: int = 4

    # Data — uses pre-extracted features: (9 frames x 12 players x 2048)
    task_type: str = "features"
    input_mode: str = "features"
    input_size: int = 2048

    # Feature paths
    features_dir: str = os.environ.get("VOLLEYBALL_FEATURES_DIR", "features")

    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    use_amp: bool = False
    patience: int = 12
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.1

    # Model
    num_classes: int = 8
    hidden_size: int = 512
    lstm_layers: int = 2
    dropout: float = 0.6
    feat_dropout: float = 0.2
    lstm_dropout: float = 0.3

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
    model_save_path: str = "checkpoints/b7/best_model_b7.pth"
    cm_save_path: str = "checkpoints/b7/confusion_matrix_b7.png"
