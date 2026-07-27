"""STUB — leakage-safe train/val/test splitting.

PORT `make_splits` from dr_rrc_replication/src/prepare_data.py (at 4591303), keeping its LOGIC
AND ALL THREE HARD ASSERTIONS INTACT. That function is independently correct and independently
justified here; it is one of the few things in experiment 1 that survives the change of model,
data source and training recipe. Copy it, do not reimplement it from memory.

TWO MECHANICAL CHANGES ARE UNAVOIDABLE — a literally verbatim copy will not run:
  1. Config lookups. The original reads `cfg["split_mode"]` and `cfg["split_ratios"]` flat; here
     they are nested (`sft_data.split_mode`, `sft_data.split_ratios`). Use
     `config_utils.resolve(cfg, "sft_data.split_mode")`. A verbatim copy KeyErrors immediately.
  2. Signature. The original takes and returns pandas DataFrames and returns a 4-tuple. `pandas`
     is a base dependency here, so keeping the DataFrame form is fine and is the smaller diff —
     the stub signature below is a placeholder, match whichever you actually port.
Change nothing else, and above all do not weaken the assertions.

What it guarantees
    - Splits on SCENARIO KEY, never on story rows. All wording variants of a scenario, and the
      easy and hard versions of the same scenario, stay on ONE side of the split. Agent easy and
      hard cases share titles, so row-level splitting leaks real content.
    - Hard-asserts no scenario-group overlap across splits. Do not weaken those assertions.
    - Two modes: `random` (stratified, roughly balanced test set) and `stakes_generalisation`
      (train on easy/low-stakes, test on hard/high-stakes — here same-scenario easy-in-train /
      hard-in-test is INTENTIONAL, it is the generalisation experiment).
    - The test set is built from the UNFILTERED held-out vignettes, so test accuracy is a real
      number rather than an artifact of the judge having removed the hard cases.

INSTRUCTION-MIX EXAMPLES MUST BYPASS THIS ENTIRELY
    The public instruction data mixed into SFT (see instruction_mix.py) has no scenario key and
    no vignette identity. Feeding it through scenario-key splitting would either crash the
    assertions or, worse, create a garbage key that silently defeats them. Instruction examples
    are appended to the TRAIN split only, after the RRC split is computed and asserted — they
    must never reach val or test, or the eval measures instruction-following rather than RRC.

TODO
    - make_splits(pool, full, cfg) — ported.
    - build_sft_records(traces, cfg) -> jsonl records, target from data_utils.build_completion,
      messages from:
          spec = None if resolve(cfg, "sft_data.strip_spec") else load_spec(cfg)[0]
          data_utils.build_messages(story, spec=spec)
      WIRE THE FLAG. Hardcoding `spec=None` makes `strip_spec: false` a silent no-op — exactly
      the class of dead config key that set_by_path's must-already-exist rule exists to prevent.
      `strip_spec: true` is the real setting (context distillation); `false` is the ablation that
      checks whether the model can do it at all with the spec present.
    - Token-length report: print p95 rendered length against cfg.model.max_seq_len. If p95
      exceeds it, RAISE max_seq_len rather than truncating — see the invariant in
      training/sft_dataset.py.
"""

from __future__ import annotations


def make_splits(pool: list[dict], full: list[dict], cfg: dict) -> dict:
    raise NotImplementedError


def build_sft_records(traces: list[dict], cfg: dict) -> list[dict]:
    raise NotImplementedError
