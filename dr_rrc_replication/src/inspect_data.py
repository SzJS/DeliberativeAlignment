"""Sanity-inspect the RRC_experiments result workbooks.

Prints sheet names, shapes, columns, and the DeepSeek-R1 RRC accuracy==1 counts
per dataset. Run this after download_data.py and before prepare_data.py to
confirm the expected layout (sheets named "{model} {approach}").

Usage:  python src/inspect_data.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dr_rrc_replication.src.config_utils import load_config
from dr_rrc_replication.src.datasets_common import RESULT_FILES, rrc_sheet_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = load_config(ap.parse_args().config)
    results = Path(cfg["data_dir"]) / "results"

    total_accurate = 0
    for domain, difficulty, fname in RESULT_FILES:
        path = results / fname
        if not path.exists():
            print(f"[inspect] MISSING: {path}")
            continue
        xls = pd.ExcelFile(path)
        print("=" * 72)
        print(f"{fname}  (domain={domain}, difficulty={difficulty})")
        print("  sheets:", xls.sheet_names)
        sheet = rrc_sheet_name("DeepSeek-R1")
        if sheet not in xls.sheet_names:
            print(f"  [WARN] expected sheet {sheet!r} not found")
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        acc1 = int((df["accuracy"] == 1).sum())
        total_accurate += acc1
        print(f"  sheet={sheet!r} shape={df.shape} cols={list(df.columns)}")
        print(
            f"  accuracy==1: {acc1}/{len(df)} | "
            f"pred={sorted(df['contractualist.prediction'].dropna().unique().tolist())} | "
            f"output={df['output'].astype(str).str.strip().str.upper().value_counts().to_dict()}"
        )
    print("=" * 72)
    print(f"TOTAL DeepSeek-R1 RRC accuracy==1 across datasets: {total_accurate}  (expected ~319)")


if __name__ == "__main__":
    main()
