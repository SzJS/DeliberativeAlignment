"""STUB — the public instruction-tuning data blended into the SFT stage (MSM's AFT recipe).

Why this exists
    SDF runs on an INSTRUCT model (`granite-4.1-8b`), and midtraining on a narrow synthetic corpus
    degrades its general coherence. MSM hit the same thing and fixed it the same way: their AFT
    stage trains on a mixture of spec-aligned data AND standard public instruction-tuning data,
    2M tokens of it, "mostly to fix incoherence caused by midtraining on Instruct models"
    (Appendix B.3). This module supplies that second half — their Table 2 verbatim.

    Secondarily it keeps the RRC signal from being the only thing SFT sees: the RRC set is a few
    hundred vignettes, so without a mix the fine-tune is narrow enough to damage everything else.

    Note the split of responsibilities: the SDF corpus stays PURE next-token prediction over
    synthetic documents (that is the mechanism MSM claims installs knowledge, and injecting chat
    formatting there would defeat it). Instruction data enters at SFT only. A separate knob,
    cfg.sdf.replay, mixes generic *raw text* into SDF to counter catastrophic forgetting — that
    is a different thing and lives in training/sdf_dataset.py.

Inputs   cfg.sft_data.instruction_mix: {scale, sources[], split, filter{enabled, oversample}}
         cfg.generation.models.instruction_filter
Outputs  a list of SFT records in the same shape splits.py produces for RRC traces.

Invariants
    - BOTH eval arms that involve SFT (`sdf_sft`, `sft_only`) get the IDENTICAL instruction
      mixture, sampled with the same seed. If the mixtures differ, the comparison measures
      instruction data rather than SDF.
    - Instruction records go into the TRAIN split only, never val or test (see splits.py).
    - Same chat template and same loss masking as RRC records: prompt masked, response
      supervised. They differ in content, not in mechanics.
    - Deterministic subsampling given cfg.seed.
    - Backfill after filtering draws from the SAME source, or the realised mixture silently drifts
      from Table 2's proportions. Record realised per-source counts in the manifest.

TODO
    - load(cfg) -> list[SftRecord]. For each entry in cfg...instruction_mix.sources: load
      (path, config, split), take n * scale deterministically under cfg.seed, normalise to our
      record shape. Guarantee >= 1 per source whenever scale > 0, or smoke runs drop the small
      sources (LongAlign at 216) to zero and stop exercising their schema handling.
    - The sources have heterogeneous schemas — `messages` on smoltalk and No Robots,
      `conversations` on some others. Normalise on the way in; do not push that into
      sft_dataset.py, which should see one shape.
    - MULTI-TURN MAPPING, a real decision to make explicit: `messages` = every turn up to and
      including the final user turn, `target` = the final assistant turn. Intermediate assistant
      turns are therefore masked as prompt. No Robots and LongAlign both contain multi-turn.
    - Spec-misalignment filter (MSM B.3): drop toxic samples, samples where the assistant
      identifies as another model ("I'm GPT-4"), and "as an AI I have no subjective
      opinions/preferences". Via jobs.run_job with the response cache, using
      cfg.generation.models.instruction_filter. Distinct from judge.py, which grades RRC CoTs
      against the rubric; this shares only the run_job plumbing.
    - Drop examples that exceed cfg.model.max_seq_len rather than truncating (same invariant as
      RRC records), and report that count separately from the RRC drop count.
    - `GAIR/lima` is GATED: it needs HF_TOKEN and a one-time terms acceptance. Fail with that
      instruction rather than a bare 401. scripts/preflight.sh probes it first.
"""

from __future__ import annotations


def load(cfg: dict) -> list[dict]:
    raise NotImplementedError
