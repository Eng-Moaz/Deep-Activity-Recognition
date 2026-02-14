"""Reproducibility utilities: seeding and run metadata."""

import json
import os
import random
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _get_git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def dump_run_metadata(cfg, output_dir: str) -> str:
    """Save run metadata (config + git SHA + timestamp) to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _get_git_sha(),
        "config": asdict(cfg),
    }

    path = os.path.join(output_dir, "run_metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Run metadata saved to: {path}")
    return path
