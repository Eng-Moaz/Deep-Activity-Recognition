import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Experiment
    experiment_name: str = "baseline8"

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
    epochs: int = 35
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    optimizer: str = "AdamW"
    use_amp: bool = False
    patience: int = 15
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.1

    # Model
    num_classes: int = 8
    hidden_size_player: int = 1024
    hidden_size_frame: int = 1024
    hidden_fc1: int = 512
    hidden_fc2: int = 256
    hidden_fc3: int = 128
    dropout: float = 0.5

    class_names: List[str] = field(default_factory=lambda: [
        'l_pass', 'r_pass',
        'l_spike', 'r_spike',
        'l_set', 'r_set',
        'l_winpoint', 'r_winpoint',
    ])

    # Class weights (inverse frequency, boosting winpoint classes)
    class_weights: List[float] = field(default_factory=lambda: [
        1.0, 1.0,   # l_pass, r_pass
        1.0, 1.0,   # l_spike, r_spike
        1.0, 1.0,   # l_set, r_set
        2.0, 2.5,   # l_winpoint, r_winpoint
    ])

    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = "ReduceLROnPlateau"
    scheduler_patience: int = 3
    gamma: float = 0.1

    # Reproducibility
    seed: int = 42

    # Paths
    model_save_path: str = "checkpoints/b8/best_model_b8.pth"
    cm_save_path: str = "checkpoints/b8/confusion_matrix_b8.png"
