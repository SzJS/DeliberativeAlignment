#!/usr/bin/env bash
# End-to-end pipeline. Assumes deps are installed (see README):
#   uv sync --extra gpu            # + optionally: --extra unsloth
# Usage: bash scripts/run_all.sh [config.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
# The src/ scripts import the absolute package `dr_rrc_replication.src.*`, so the
# repo root (parent of this dir) must be importable. Put it on PYTHONPATH here
# rather than relying on the caller's environment.
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="${1:-config.yaml}"

uv run python src/download_data.py --config "$CONFIG"
uv run python src/inspect_data.py  --config "$CONFIG"
uv run python src/prepare_data.py  --config "$CONFIG"
uv run python src/train.py --check --config "$CONFIG"   # masking sanity check
uv run python src/train.py         --config "$CONFIG"
uv run python src/evaluate.py      --config "$CONFIG"

echo "Done. See artifacts/metrics.json"
