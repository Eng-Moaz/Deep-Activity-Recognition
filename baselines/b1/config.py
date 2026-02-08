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

    # Paths
    model_save_path: str = "models/b1/best_model.pth"
    cm_save_path: str = "models/b1/confusion_matrix.png"