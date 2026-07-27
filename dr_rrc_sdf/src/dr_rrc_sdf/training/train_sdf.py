"""STUB — SDF continued pretraining (pipeline stage 2 of 5).

Usage
    uv run python -m dr_rrc_sdf.training.train_sdf --check --config config.yaml   # inspect, then EXIT
    uv run python -m dr_rrc_sdf.training.train_sdf --config config.yaml
    accelerate launch -m dr_rrc_sdf.training.train_sdf --config config.yaml      # multi-GPU / FSDP

Contract
    Plain next-token prediction over the packed corpus. `Trainer` +
    DataCollatorForLanguageModeling(mlm=False) — NOT trl's SFTTrainer, which exists to do the
    prompt/completion masking this stage must not do.

Inputs   sdf_dataset.build_dataset, cfg.sdf_train
Outputs  artifacts/models/sdf/{adapter or full weights}/ + trainer/ checkpoints

--check (run this before any real training)
    Decode one packed block and print it, then assert `labels == input_ids` except at padding.
    The printed block MUST look like raw document prose and MUST NOT contain <|start_of_role|>.

    This is the exact INVERSE of train_sft.py --check, which asserts that only the assistant span
    is unmasked. Two stages, two opposite assertions; running both is how you prove neither stage
    has drifted into the other's mechanics.

Invariants
    - No loss mask, no chat template. See sdf_dataset.py.
    - Resumable from artifacts/models/sdf/trainer/ via resume_from_checkpoint — assume the pod
      gets preempted.
    - Log to W&B: loss, LR, grad norm, throughput, GPU/mem utilisation, and the full config.

TODO
    - full_finetune path (cfg.sdf_train.full_finetune: true) is the DEFAULT and the faithful one:
      MSM trains the corpus as pretraining data, and LoRA may lack the capacity to install
      knowledge rather than style. 8B full FT does not fit one card — wire FSDP via accelerate.
    - LoRA path as the cheap fallback (r=64 in config, higher than SFT's r=16 for the same
      reason).
    - After training, ALWAYS run merge.py. The SFT stage must initialise from merged weights.
    - GATE: run the spec_recall eval on this checkpoint (arm `sdf_only`) BEFORE paying for the
      SFT stage. If SDF installed nothing measurable, that is the moment to find out — see the
      risk in experiment_description.md.
"""

from __future__ import annotations


def check(cfg: dict) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
