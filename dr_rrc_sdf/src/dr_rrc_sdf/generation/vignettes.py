"""STUB — vignette source switch (cfg.sft_data.vignette_source: "paper" | "generated").

Contract
    `load_vignettes(cfg)` returns a uniform list of Vignette records regardless of source, so
    nothing downstream (cot.py, judge.py, splits.py, eval datasets) knows which was used.

Uniform record
    {scenario_key, domain, difficulty, story, label}
    `scenario_key` is what splits.py groups on — it is the leakage-safety mechanism, so both
    sources MUST provide a meaningful one.

TODO
    - Dispatch to vignettes_paper / vignettes_generated.
    - Validate the uniform schema at the boundary, so a source-specific bug surfaces here rather
      than three stages later.
    - Record the source and its provenance (repo sha, or generating model + spec sha) in
      artifacts/sft/stats.json — a mixed-source dataset must be detectable.
"""

from __future__ import annotations


def load_vignettes(cfg: dict) -> list[dict]:
    raise NotImplementedError
