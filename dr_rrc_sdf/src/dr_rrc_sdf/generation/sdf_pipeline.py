"""STUB — CLI entry point for SDF corpus generation (pipeline stage 1 of 5).

Usage
    uv run python -m dr_rrc_sdf.generation.sdf_pipeline --config config.yaml
    uv run python -m dr_rrc_sdf.generation.sdf_pipeline --dry-run     # render + price, NO API calls
    uv run python -m dr_rrc_sdf.generation.sdf_pipeline --emit-manifest   # stages 1-4 only

Contract
    Drive sdf.py's five stages end to end, then filter, then write artifacts/sdf/corpus.jsonl.

TODO
    - argparse: --config, --set a.b=c (config_utils.parse_set_args), --dry-run, --emit-manifest,
      --no-cache, --max-cost-usd (override).
    - Run stages 1-4, write assets/sdf_manifest.json, and STOP under --emit-manifest so the
      manifest can be reviewed before paying for stage 5. Stage 5 is ~99% of the cost.
    - Resume by re-running: cache.py makes completed work free.
    - Print a final summary: documents kept/dropped per filter reason, diversity number, spend.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
