"""Config loading with smoke-mode overrides."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("smoke"):
        overrides = cfg.get("smoke_overrides", {}) or {}
        cfg.update(overrides)
        print(f"[config] SMOKE mode: applied overrides {list(overrides)}")
    return cfg
