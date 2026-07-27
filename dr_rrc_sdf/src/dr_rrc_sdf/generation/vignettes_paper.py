"""STUB — vignettes from the RRC paper's workbooks (vignette_source: "paper").

Contract
    Clone https://github.com/mint-philosophy/RRC_experiments (MIT) into cfg.paths.data, read the
    four Vignettes_{Agent,Development}_{Easy,Hard}_Cases.xlsx workbooks, and return the uniform
    record: {scenario_key, domain, difficulty, story, label}.

This is a REDUCTION of dr_rrc_replication/src/datasets_common.py, not a copy
    That module (~155 lines) exists to mine the paper's *reasoning traces*: APPROACH_SHEET,
    sheet_name(), _iter_best_per_vignette(), load_condition() with its three conditions, plus the
    whole of framed_deliberation.py and scripts/build_framed_sidecar.py. This experiment
    GENERATES its own traces, so all of that is dead weight here. We need only the story, the
    ground-truth label (`contractualist.prediction`), and the scenario key. ~60 lines.

Invariants
    - Clone into THIS experiment's cfg.paths.data. Do NOT point at
      ../dr_rrc_replication/data/ — that directory is git-ignored, may not exist, and on RunPod
      you may have synced only one experiment folder.
    - Idempotent: skip the clone if the results dir is already present.

Known data facts (carried over from dr_rrc_replication/tasks/todo.md, still true and still
load-bearing for how results must be read)
    - Agent and development workbooks have slightly different column schemas — normalize both.
    - Scenario duplication: agent 40 scenarios x 3 wording variants, development 13 x 5.
      scenario_key must strip the variant, or wording variants leak across the split.
    - ALL hard vignettes are ground-truth YES and ALL easy are NO. So difficulty alone predicts
      the label. This is why rrc_decision must not be the headline metric — see README.

TODO
    - download(cfg): git clone --depth 1 if absent.
    - load(cfg) -> list[Vignette], normalizing both column schemas.
    - scenario_key derivation that collapses wording variants AND pairs easy/hard versions of the
      same scenario (agent easy/hard share titles — dr_rrc_replication hit real leakage here).
"""

from __future__ import annotations


def download(cfg: dict) -> None:
    raise NotImplementedError


def load(cfg: dict) -> list[dict]:
    raise NotImplementedError
