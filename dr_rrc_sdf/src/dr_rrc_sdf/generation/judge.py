"""STUB — spec-aware judge filter (deliberative alignment, step 2).

Contract
    Grade each (vignette, CoT, answer) against assets/judge_rubric.md and keep only traces whose
    reasoning applies the RRC procedure correctly. One accepted trace per vignette becomes an
    SFT target.

Inputs   artifacts/sft/cot_raw.jsonl, assets/judge_rubric.md, cfg.sft_data.judge
Outputs  artifacts/sft/judge.jsonl — one verdict per sample, ACCEPTED AND REJECTED ALIKE

Invariants
    - Answer correctness is checked DETERMINISTICALLY against the label before the judge is
      called, and is not a rubric criterion. A judge asked to grade correctness anchors every
      other score on whether the answer was right.
    - Persist rejected verdicts too. The reject reasons are the diagnostic when the accept rate
      is surprising, and they are gone forever if only accepts are written.
    - Selecting one trace per vignette from the accepted pool must be DETERMINISTIC given the
      seed, or the dataset changes silently between runs.

SPEC-LEAKAGE FILTER — no analogue in dr_rrc_replication
    CoTs here are generated WITH the spec in context and trained on with the spec STRIPPED
    (context distillation). So a CoT saying "as the policy above states" / "per the instructions
    provided" becomes a dangling reference to a document that will not exist at inference, and
    trains the model to gesture at an absent authority instead of recalling the procedure.

    dr_rrc_replication never hit this because it reused the paper's published traces rather than
    generating its own. Here it is a real and likely-common failure mode.

    Decide (see assets/judge_rubric.md): FILTER these traces (simple, costs yield) or REWRITE
    them with a cheap follow-up call (preserves yield, adds a stage). Whichever — measure how
    often it fires, because if it is most traces then the generation prompt is the thing to fix.

TODO
    - async judge(samples, cfg) -> list[JudgeVerdict], via jobs.run_job.
    - select_best(verdicts, seed) -> one trace per vignette, deterministic.
    - Report accept rate overall and per criterion in artifacts/sft/stats.json. A very high
      accept rate means the rubric is not discriminating, not that generation is excellent.
"""

from __future__ import annotations


async def judge(samples: list[dict], cfg: dict) -> list[dict]:
    raise NotImplementedError


def select_best(verdicts: list[dict], seed: int) -> list[dict]:
    raise NotImplementedError
