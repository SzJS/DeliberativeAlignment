"""STUB — SFT on the SDF checkpoint (pipeline stage 4 of 5).

Usage
    uv run python -m dr_rrc_sdf.training.train_sft --check --config config.yaml   # print mask, EXIT
    uv run python -m dr_rrc_sdf.training.train_sft --config config.yaml
    # the sft_only ablation arm:
    uv run python -m dr_rrc_sdf.training.train_sft --config config.yaml --set sft_train.init_from=base

Contract
    LoRA SFT with completion-only loss, initialised from the MERGED SDF weights (or from the raw
    base for the `sft_only` arm). trl SFTTrainer, or plain Trainer with our own collator.

--check (run before any real training)
    Print the MASKED and UNMASKED token blocks. UNMASKED must be the assistant target ONLY:
    `<reasoning>...</reasoning>\\n\\nANSWER: YES` plus eos. MASKED must be the system+user prompt.
    If prompt tokens appear in UNMASKED, stop and fix before training.

    This is the exact INVERSE of train_sdf.py --check (which asserts NOTHING is masked). Run both.

Invariants
    - cfg.sft_train.init_from == "sdf" must load artifacts/models/merged_sdf/, not base + an
      unmerged adapter. See merge.py — this is the silent-null-result trap.
    - Both SFT arms (`sdf_sft`, `sft_only`) train on the IDENTICAL dataset, including the same
      instruction mix at the same seed. Differing data would confound the comparison that is the
      entire point of running two arms.
    - Log everything to W&B, including sample generations. No untracked runs.
    - Resumable via resume_from_checkpoint.

TODO
    - check(cfg) and main()
    - Best-by-val-loss checkpointing (cfg.sft_train.save_best)
    - Learning rate: config starts at 1e-5, NOT dr_rrc_replication's 1e-4 — Granite's
      logits_scaling: 16.0 changes loss magnitude and effective LR. Sweep it.
    - After training, run merge.py for the arm being trained.
"""

from __future__ import annotations


def check(cfg: dict) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
