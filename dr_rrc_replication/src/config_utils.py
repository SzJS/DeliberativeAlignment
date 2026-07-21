"""Config loading with smoke-mode overrides."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_env() -> None:
    """Load the repo .env into os.environ. No-op if python-dotenv or .env is absent.

    Walks up from the cwd (scripts run from dr_rrc_replication/) to find the
    repo-root .env, so secrets like WANDB_API_KEY reach the process without a
    manual export. Does NOT override variables already set in the real
    environment, so an explicit `export` still wins.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


def load_config(path: str = "config.yaml") -> dict:
    load_env()
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("smoke"):
        overrides = cfg.get("smoke_overrides", {}) or {}
        cfg.update(overrides)
        print(f"[config] SMOKE mode: applied overrides {list(overrides)}")
    return cfg
