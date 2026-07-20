"""Clone the RRC_experiments repo (data source) into the configured data_dir.

Usage:  python src/download_data.py [--config config.yaml]
Idempotent: if the target already contains the results, it does nothing.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dr_rrc_replication.src.config_utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = load_config(ap.parse_args().config)

    dest = Path(cfg["data_dir"])
    if (dest / "results").is_dir():
        print(f"[download_data] already present at {dest} — skipping clone.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download_data] cloning {cfg['data_repo_url']} -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", cfg["data_repo_url"], str(dest)],
        check=True,
    )
    print("[download_data] done.")


if __name__ == "__main__":
    main()
