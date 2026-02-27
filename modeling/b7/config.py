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

    # Data — uses pre-extracted CNN features: (9 frames x 12 players x 2048)
    task_type: str = "features"
    input_mode: str = "features"
    input_size: int = 2048  # Raw CNN features
    feature_file_stem: str = "features"

    # Feature paths
    features_dir: str = os.environ.get("VOLLEYBALL_FEATURES_DIR", "features")

    # Training
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 4e-4
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    use_amp: bool = False
    patience: int = 15
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.0

    # Model
    num_classes: int = 8
    hidden_size_player: int = 512  # LSTM1: player-level
    hidden_size_scene: int = 512   # LSTM2: scene-level
    dropout: float = 0.5

    class_names: List[str] = field(default_factory=lambda: [
        'l_pass', 'r_pass',
        'l_spike', 'r_spike',
        'l_set', 'r_set',
        'l_winpoint', 'r_winpoint',
    ])

    # Scheduler
    use_scheduler: bool = False

    # Reproducibility
    seed: int = 42

    # Paths
    model_save_path: str = "checkpoints/b7/best_model_b7.pth"
    cm_save_path: str = "checkpoints/b7/confusion_matrix_b7.png"
