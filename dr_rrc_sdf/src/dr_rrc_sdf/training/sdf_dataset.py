"""STUB — the SDF corpus as packed next-token blocks.

THIS IS CONTINUED PRETRAINING, NOT SFT. IF ANY TOKEN IS MASKED, THE STAGE IS WRONG.

Contract
    artifacts/sdf/corpus.jsonl -> fixed-length token blocks with `labels == input_ids`.

    No chat template. No role markers. No instruction format. No prompt/completion split. No
    loss mask. The documents are trained exactly like pretraining data, because that is MSM's
    entire claim: install knowledge about the assistant through the same mechanism the model used
    to acquire world knowledge. Formatting the corpus as conversations would defeat it — and the
    chat format is installed later, by the SFT stage, which is where it belongs.

Inputs   artifacts/sdf/corpus.jsonl, cfg.sdf_train.{block_size, packing}, cfg.sdf.replay
Outputs  a torch/HF dataset of {input_ids, labels, attention_mask}

Invariants
    - labels == input_ids everywhere except padding. This is the inverse of the SFT stage's
      invariant, and stating both side by side is how the two stages are kept from contaminating
      each other. train_sdf.py --check asserts it.
    - Documents are concatenated with an eos separator and packed to block_size, so the GPU never
      trains on padding. Packing is the difference between a compute-bound and an
      embarrassingly-idle run.
    - Padding uses <|pad|>, distinct from eos (see data_utils.set_pad_token). With pad == eos the
      packed stage cannot tell padding from a genuine document boundary.

REPLAY MIX — cfg.sdf.replay
    Continued pretraining on a narrow synthetic corpus causes catastrophic forgetting and general
    degeneration. The standard mitigation is to blend in generic pretraining text (suggested
    20-50%). This is a config knob from day one rather than a retrofit, because discovering the
    need for it after a bad run costs a full regeneration and retrain.

    Distinct from cfg.sft_data.instruction_mix: that is INSTRUCTION data in the SFT stage; this is
    RAW TEXT here. Do not merge them.

TODO
    - build_dataset(cfg, tokenizer) -> Dataset
    - pack(token_streams, block_size) with an eos separator between documents
    - Interleave the replay dataset at cfg.sdf.replay.fraction, streamed rather than materialised
    - Report token counts: corpus tokens, replay tokens, blocks, and the effective epoch size
"""

from __future__ import annotations


def build_dataset(cfg: dict, tokenizer):
    raise NotImplementedError


def pack(token_streams, block_size: int):
    raise NotImplementedError
