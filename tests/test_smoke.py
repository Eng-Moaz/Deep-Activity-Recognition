"""Smoke tests — verify that configs, models, and training engine import and instantiate correctly."""

import torch
import pytest


def test_b1_config_imports():
    from modeling.b1.config import Config
    cfg = Config()
    assert cfg.task_type == "scene"
    assert cfg.input_mode == "scenefull"
    assert cfg.seed == 42
    assert hasattr(cfg, "videos_dir")
    assert hasattr(cfg, "tracks_dir")


def test_b3_config_imports():
    from modeling.b3.config import Config_stg1, Config_stg2
    cfg1 = Config_stg1()
    cfg2 = Config_stg2()
    assert cfg1.task_type == "player"
    assert cfg1.input_mode == "action_train"
    assert cfg2.task_type == "scene"
    assert cfg2.input_mode == "scenecrops"


def test_b4_config_imports():
    from modeling.b4.config import Config
    cfg = Config()
    assert cfg.task_type == "scene"
    assert cfg.input_mode == "scenefull_temporal"


def test_b5_config_imports():
    from modeling.b5.config import Config_stg1, Config_stg2
    cfg1 = Config_stg1()
    cfg2 = Config_stg2()
    assert cfg1.task_type == "player"
    assert cfg1.input_mode == "temporal"
    assert cfg2.task_type == "scene"
    assert cfg2.input_mode == "scenecrops_temporal"


def test_b6_config_imports():
    from modeling.b6.config import Config
    cfg = Config()
    assert cfg.task_type == "scene"
    assert cfg.input_mode == "scenecrops_temporal"
    assert cfg.num_classes == 8
    assert cfg.num_classes_stg1 == 9
    assert cfg.hidden_size == 256
    assert hasattr(cfg, "saved_resnet50_path")


def test_b6_model_builds(tmp_path):
    import torch
    from modeling.b3.model import Baseline3_stg1
    from modeling.b6.config import Config
    from modeling.b6.model import Baseline6

    # Create a temporary B3 Stage 1 checkpoint so Baseline6 can load it
    stg1_cfg = type('Cfg', (), {'num_classes': 9, 'dropout': 0.5})()
    stg1_model = Baseline3_stg1(stg1_cfg)
    ckpt_path = str(tmp_path / "b3_stg1.pth")
    torch.save(stg1_model.state_dict(), ckpt_path)

    cfg = Config(saved_resnet50_path=ckpt_path)
    model = Baseline6(cfg)
    model.eval()

    # Dummy input: (batch=2, seq=9, players=12, C=3, H=224, W=224)
    x = torch.randn(2, 9, 12, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8)


def test_b1_model_builds():
    from modeling.b1.config import Config
    from modeling.b1.model import Baseline1
    cfg = Config()
    model = Baseline1(cfg)
    assert model is not None
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 8)


def test_b3_stg1_model_builds():
    from modeling.b3.config import Config_stg1
    from modeling.b3.model import Baseline3_stg1
    cfg = Config_stg1()
    model = Baseline3_stg1(cfg)
    assert model is not None
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 9)


def test_training_engine_imports():
    from training.engine import train_one_epoch, validate, run_training
    assert callable(train_one_epoch)
    assert callable(validate)
    assert callable(run_training)


def test_reproducibility_imports():
    from training.reproducibility import set_seed, dump_run_metadata
    assert callable(set_seed)
    assert callable(dump_run_metadata)


def test_set_seed_runs():
    from training.reproducibility import set_seed
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_evaluation_imports():
    from helper_utils.evaluation import evaluate_test_set
    assert callable(evaluate_test_set)


def test_dataloader_imports():
    from data_utils.dataloader import get_data_loaders
    assert callable(get_data_loaders)


def test_dataset_imports():
    from data_utils.dataset import VolleyballSceneDataset, VolleyballPlayerDataset
    assert VolleyballSceneDataset is not None
    assert VolleyballPlayerDataset is not None
