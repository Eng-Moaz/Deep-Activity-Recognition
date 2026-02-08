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

    # Training
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.001

    # Model
    num_classes: int = 9
    dropout: float = 0.5
    class_names: List[str] = field(default_factory=lambda: ['blocking', 'digging', 'falling', 'jumping',
                    'moving', 'setting', 'spiking', 'standing', 'waiting'])

    # Scheduler
    use_scheduler: bool = True
    step_size: int = 10
    gamma: float = 0.1

    # Paths
    model_save_path: str = "models/b3/best_model_b3_stg1.pth"
    cm_save_path: str = "models/b3/confusion_matrix_b3_stg1.png"

@dataclass
class Config_stg2(Config_stg1):
    # Experiment
    experiment_name: str = "baseline3_stage2"

    #Training
    epochs: int = 12
    learning_rate: float = 5e-4
    batch_size: int = 8

    # Model
    num_classes: int = 8
    class_names: List[str] = field(default_factory=lambda: [
        'l_pass', 'r_pass',
        'l_spike', 'r_spike',
        'l_set', 'r_set',
        'l_winpoint', 'r_winpoint'
    ])

    # Paths
    saved_resnet50_path: str = "models/b3/best_model_b3_stg1.pth"
    model_save_path: str = "models/b3/best_model_b3_stg2.pth"
    cm_save_path: str = "models/b3/confusion_matrix_b3_stg2.png"