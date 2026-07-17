"""YAML config loading/merging.

Configs are plain dicts.  A solution's ``config.yaml`` may set ``extends`` to a
path (relative to itself) to inherit a base config; leaf keys override.
No magic numbers in code — model names, top_k, thresholds, device all live here.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: "os.PathLike | str") -> Dict[str, Any]:
    """Load a YAML config, resolving a single ``extends: <relative path>``."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config((path.parent / parent).resolve())
        cfg = _deep_merge(base, cfg)
    return cfg
