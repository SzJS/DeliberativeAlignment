"""STUB — inspect Datasets built from the frozen test split and from the spec.

Contract
    Convert our jsonl artifacts into inspect `Sample`s, carrying metadata that the scorers group
    on.

TODO
    - rrc_dataset(cfg) — from artifacts/sft/test.jsonl.
      Sample(input=story, target="YES"|"NO",
             metadata={scenario_key, domain, difficulty, source})
      The metadata keys are what grouped(accuracy(), "difficulty") reads, and the per-slice
      breakdown is the only way to see whether the hard cases are carrying the result.
    - spec_recall_dataset(cfg) — probes derived from assets/rrc_spec.md. Three kinds:
      free recall ("what should you do when stakes are high and the case is unusual?"),
      multiple choice, and clause attribution. These are generated once and COMMITTED (to
      assets/), not regenerated per run — a moving eval set makes runs incomparable.
    - Never build eval samples from the train or val split. The test split is frozen at
      sft_pipeline time and is the only thing any arm sees.
"""

from __future__ import annotations


def rrc_dataset(cfg: dict):
    raise NotImplementedError


def spec_recall_dataset(cfg: dict):
    raise NotImplementedError
