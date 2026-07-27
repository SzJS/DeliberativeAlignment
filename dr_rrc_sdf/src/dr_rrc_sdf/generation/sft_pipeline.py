"""STUB — CLI entry point for SFT data generation (pipeline stage 3 of 5).

Usage
    uv run python -m dr_rrc_sdf.generation.sft_pipeline --config config.yaml
    uv run python -m dr_rrc_sdf.generation.sft_pipeline --dry-run    # render + price, NO API calls

Contract
    vignettes -> CoT generation -> judge filter -> select one trace per vignette -> blend the
    instruction mix into train -> leakage-safe splits -> artifacts/sft/{train,val,test}.jsonl
    + splits.json.

TODO
    - argparse: --config, --set a.b=c, --dry-run, --no-cache.
    - Order matters: compute and ASSERT the RRC scenario-key split FIRST, then append the
      instruction mix to train. Appending before splitting breaks the leakage assertions.
    - Print the stage summary dr_rrc_replication's prepare_data prints, since it is what
      VERIFICATION checks: pool size after malformed-drop, train/val/test counts, test label base
      rate and majority-class accuracy, per-(domain, difficulty) counts, p95 rendered token
      length vs max_seq_len, judge accept rate, and the instruction-mix count as a SEPARATE
      number from the RRC count.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
