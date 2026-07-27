"""Config loading: repo .env, sectioned YAML, dotted-path smoke overrides, resolved snapshot.

Ported from dr_rrc_replication/src/config_utils.py at 4591303; intentionally divergent — do not
re-sync. `load_env` is verbatim. `load_config` is rewritten: that version applied smoke
overrides with a flat ``cfg.update(overrides)``, which cannot reach into a per-stage section
(``sdf_train.epochs``) and which silently *adds* a mistyped key instead of raising.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()


def load_env() -> None:
    """Load the repo .env into os.environ. No-op if python-dotenv or .env is absent.

    Walks up from the cwd (scripts run from dr_rrc_sdf/) to find the repo-root .env, so secrets
    reach the process without a manual export: OPENROUTER_API_KEY for both generation pipelines
    and for inspect_ai's `openrouter/` provider, and WANDB_API_KEY for run logging. Does NOT
    override variables already set in the real environment, so an explicit `export` still wins.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


def _split_path(dotted: str) -> list[str]:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise ValueError(f"empty config path: {dotted!r}")
    return parts


def resolve(cfg: dict, dotted: str, default: Any = _MISSING) -> Any:
    """Read a nested config value by dotted path, so stage code never chains ``cfg["a"]["b"]``."""
    node: Any = cfg
    for part in _split_path(dotted):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"config path not found: {dotted!r}")
            return default
        node = node[part]
    return node


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    """Set a nested config value by dotted path. The path must already exist.

    Requiring the key to exist is the point: a flat ``cfg.update()`` accepts ``epocs: 1`` and
    does nothing, so a smoke run silently trains for the real number of epochs. Here that
    raises.
    """
    parts = _split_path(dotted)
    node: Any = cfg
    for i, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"override path {dotted!r} does not exist (missing {'.'.join(parts[: i + 1])!r})")
        node = node[part]
    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node:
        raise KeyError(f"override path {dotted!r} does not exist (missing leaf {leaf!r})")
    node[leaf] = value


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> list[str]:
    """Apply a mapping of dotted path -> value. Returns the applied paths, for logging."""
    for dotted, value in overrides.items():
        set_by_path(cfg, dotted, value)
    return list(overrides)


def config_sha256(cfg: dict) -> str:
    """Stable hash of the fully-resolved config.

    THE ONE CONFIG SHA. It is stored as ``_resolved_sha256`` inside the config dict and inside
    the snapshot file, and `scripts/lib.sh` READS it from there rather than hashing the file —
    hashing the snapshot would give a second, different value (the snapshot contains the key),
    and two numbers under one name is how stamps and artifact manifests end up silently
    incomparable. Record this value, not a hash of anything else.

    ``_resolved_sha256`` is excluded so the hash is stable across re-loads of the same config.
    """
    blob = yaml.safe_dump(
        {k: v for k, v in cfg.items() if k != "_resolved_sha256"},
        sort_keys=True,
        default_flow_style=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def load_config(
    path: str = "config.yaml",
    cli_overrides: dict[str, Any] | None = None,
    snapshot: bool = True,
) -> dict:
    """Load config.yaml, apply smoke overrides then CLI overrides, and snapshot the result.

    ``cli_overrides`` comes from ``--set a.b.c=value`` on any stage entry point and is applied
    after smoke overrides, so it always wins.

    With ``snapshot=True`` the fully-resolved config is written to
    ``<paths.artifacts>/config.resolved.yaml`` — every stage then records exactly what it ran
    with, and ``scripts/lib.sh`` compares hashes to catch a resumed stage whose config moved.
    """
    load_env()
    cfg = yaml.safe_load(Path(path).read_text())

    if cfg.get("smoke", {}).get("enabled"):
        applied = apply_overrides(cfg, cfg["smoke"].get("overrides", {}) or {})
        print(f"[config] SMOKE mode: applied {len(applied)} overrides: {applied}")

    if cli_overrides:
        applied = apply_overrides(cfg, cli_overrides)
        print(f"[config] CLI overrides: {applied}")

    sha = config_sha256(cfg)
    cfg["_resolved_sha256"] = sha

    if snapshot:
        out = Path(resolve(cfg, "paths.artifacts", "artifacts"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.resolved.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)
        )
    print(f"[config] resolved sha256: {sha[:12]}")
    return cfg


def parse_set_args(pairs: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--set a.b.c=value`` into a dict, YAML-decoding each value so
    ``--set sdf_train.epochs=1`` gives an int and ``--set eval.limit=null`` gives None."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out
