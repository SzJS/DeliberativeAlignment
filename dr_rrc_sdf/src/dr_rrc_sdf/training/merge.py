"""STUB — merge a LoRA adapter into its base weights. LOAD-BEARING, not a convenience.

Usage
    uv run python -m dr_rrc_sdf.training.merge --stage sdf --config config.yaml
    uv run python -m dr_rrc_sdf.training.merge --stage sft --arm sdf_sft --config config.yaml

Contract
    Produce a plain, self-contained HF checkpoint directory for a training stage's output.

    TWO PATHS, and the full-FT one is the DEFAULT (cfg.sdf_train.full_finetune: true):
      - LoRA output   -> peft `merge_and_unload()` into the base, write the result.
      - full-FT output-> there is NO adapter to merge. The trainer's own output directory already
                         holds complete weights; this stage normalises it into the expected
                         `merged_*` location (copy, or hardlink/symlink if disk is tight) and
                         ensures the tokenizer + chat template are present.
    Detect which by looking for an adapter_config.json, and do NOT assume peft.

Why it is load-bearing
    1. SFT initialisation. With cfg.sft_train.init_from == "sdf", the SFT stage loads
       artifacts/models/merged_sdf/. Stacking a second LoRA on top of an UNMERGED first adapter
       silently trains against the BASE model — the SDF stage would contribute nothing, the run
       would complete cleanly, and you would report a null result for SDF that was actually a
       plumbing bug. There is no error message for this; the merge is the only thing preventing
       it.
    2. Eval serving. inspect_ai can serve `vllm/BASE:adapter` for a single adapter, but the
       `sdf_sft` arm is two stacked adaptations. Merged checkpoints keep every arm on one code
       path.

Outputs
    artifacts/models/merged_sdf/          (base + SDF adapter)      -> SFT init, `sdf_only` arm
    artifacts/models/merged_sdf_sft/      (merged_sdf + SFT adapter) -> `sdf_sft` arm
    artifacts/models/merged_sft_only/     (base + SFT adapter)       -> `sft_only` arm

Invariants
    - Copy the tokenizer AND the installed chat template into every merged directory, or the
      served model has no template and inspect's chat solvers produce raw concatenated text.
    - Merge in bf16 to match the training dtype.
    - Record in the output dir which checkpoints were merged, with the config sha, so an arm's
      provenance is recoverable from disk rather than from memory.

TODO
    - merge(base_path, adapter_path, out_path) for the LoRA path
    - promote(train_out_path, out_path) for the full-FT path
    - CLI with --stage {sdf,sft} / --arm {sdf_sft,sft_only}; skip if the output exists and is
      newer than its inputs. Called as its own pipeline stage (`merge_sdf`, `merge_sft` in
      scripts/run_all.sh) — it is NOT a step tacked onto the end of the trainers, so that a
      re-merge doesn't require a re-train.
    - Disk check first: three merged 8B bf16 copies is ~48GB
"""

from __future__ import annotations


def merge(base_path: str, adapter_path: str, out_path: str) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
