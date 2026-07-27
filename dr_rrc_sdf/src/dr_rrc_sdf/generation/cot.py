"""STUB — RRC reasoning-trace generation (deliberative alignment, step 1).

Contract
    For each vignette, prompt a strong model with the RRC SPEC IN CONTEXT and ask it to reason
    over the spec and answer. Sample n times at temperature > 0 for rejection-sampling headroom,
    since judge.py will reject a large fraction.

Inputs   vignettes (from vignettes.py), spec (from spec.py), cfg.sft_data.cot
Outputs  artifacts/sft/cot_raw.jsonl — {vignette_id, sample_idx, reasoning, answer, well_formed}

Invariants
    - Build the prompt with data_utils.build_messages(story, spec=spec). Do NOT hand-roll a
      second message constructor: the SFT input, the eval input, and this generation prompt must
      stay in lockstep, and the way that breaks is someone adding a "just for generation"
      variant.
    - Parse with data_utils.parse_output — the one parser, every source.
    - n_samples_per_vignette > 1 at temperature > 0. Greedy single samples give the judge nothing
      to select from, and the accept rate then determines the dataset size rather than the
      quality bar doing so.

TODO
    - async generate(vignettes, spec, cfg) -> list[CotSample], via jobs.run_job.
    - Record which samples were well_formed vs salvaged; a high salvage rate means
      FORMATTING_INSTRUCTIONS needs work, not that the model is bad at ethics.
    - Do NOT filter here. Generation and filtering stay separate stages so the accept rate is
      measurable and the rubric can be changed without regenerating (the cache makes the rerun
      free only if the generation prompt is untouched).
"""

from __future__ import annotations


async def generate(vignettes: list[dict], spec: str, cfg: dict) -> list[dict]:
    raise NotImplementedError
