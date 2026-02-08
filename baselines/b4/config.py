from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Experiment
    experiment_name: str = "baseline4_temporal"

    # System
    device: str = "cuda"
    num_workers: int = 4

    # Data
    data_root: str = "/kaggle/input/volleyball/volleyball_"

    # Training
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 0.0001
    weight_decay: float = 0.001

    # Model (LSTM Specifics)
    num_classes: int = 8
    hidden_size: int = 256  # Size of LSTM hidden state
    lstm_layers: int = 1  # Number of stacked LSTMs
    dropout: float = 0.5

    class_names: List[str] = field(default_factory=lambda: [
        'l_pass', 'r_pass',
        'l_spike', 'r_spike',
        'l_set', 'r_set',
        'l_winpoint', 'r_winpoint'
    ])

    # Scheduler
    use_scheduler: bool = True
    step_size: int = 10
    gamma: float = 0.1

    # Paths
    trained_resnet_path: str = "/kaggle/working/Deep-Activity-Recognition/models/b1/best_model.pth"
    model_save_path: str = "models/b4/best_model_b4.pth"
    cm_save_path: str = "models/b4/confusion_matrix_b4.png"
