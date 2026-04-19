"""Config loader — merges base.yaml + stage-specific yaml. [ENG]"""
import os
import yaml
from pathlib import Path


def load_config(stage_config_path: str) -> dict:
    root = Path(__file__).parent.parent
    base_path = root / "configs" / "base.yaml"

    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    with open(stage_config_path) as f:
        stage_cfg = yaml.safe_load(f)

    # strip 'defaults' key before merging
    stage_cfg.pop("defaults", None)
    cfg = _deep_merge(cfg, stage_cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_device(cfg: dict):
    import torch
    requested = cfg.get("training", {}).get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
