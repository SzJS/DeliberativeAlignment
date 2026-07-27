"""STUB — the public instruction-tuning data blended into the SFT stage (MSM's AFT recipe).

Why this exists
    We start from a BASE model with no instruction-following ability at all, and the RRC set is
    only a few hundred vignettes — nowhere near enough to install general instruction behaviour.
    MSM's AFT stage trains on a mixture of spec-aligned data AND standard public instruction-
    tuning data; this module supplies the second half.

    Note the split of responsibilities: the SDF corpus stays PURE next-token prediction over
    synthetic documents (that is the mechanism MSM claims installs knowledge, and injecting chat
    formatting there would defeat it). Instruction data enters at SFT only. A separate knob,
    cfg.sdf.replay, mixes generic *raw text* into SDF to counter catastrophic forgetting — that
    is a different thing and lives in training/sdf_dataset.py.

Inputs   cfg.sft_data.instruction_mix: {dataset, split, n_examples, ratio}
Outputs  a list of SFT records in the same shape splits.py produces for RRC traces.

Invariants
    - BOTH eval arms that involve SFT (`sdf_sft`, `sft_only`) get the IDENTICAL instruction
      mixture, sampled with the same seed. If the mixtures differ, the comparison measures
      instruction data rather than SDF.
    - Instruction records go into the TRAIN split only, never val or test (see splits.py).
    - Same chat template and same loss masking as RRC records: prompt masked, response
      supervised. They differ in content, not in mechanics.
    - Deterministic subsampling given cfg.seed.

TODO
    - PICK THE DATASET. Open question — needs a licence check and a size decision. Candidates:
      tulu-3-sft-mixture, OpenHermes-2.5, SmolTalk. Record the choice and the licence in
      README.md under data provenance, alongside the RRC_experiments MIT note.
    - Decide the ratio. Too little and the model never learns to follow instructions; too much
      and the RRC signal is diluted into noise. This wants a small sweep, not a guess.
    - load(cfg) -> list[SftRecord]: stream the dataset, subsample to n_examples (or `ratio` x
      the RRC train count if set), convert to our message format via
      data_utils.build_messages/build_completion where the schema allows.
    - Drop examples that exceed cfg.model.max_seq_len rather than truncating (same invariant as
      RRC records).
"""

from __future__ import annotations


def load(cfg: dict) -> list[dict]:
    raise NotImplementedError
