"""STUB — content-addressed disk cache for LLM responses. This IS the resumability mechanism.

Contract
    Wrap any generation call so a re-run reuses completed work. A crash at document 4,000 of
    5,000 costs only the in-flight requests, not the run.

Inputs
    A request (model, messages, sampling params, stage, item id).

Outputs
    outputs/cache/<stage>/<sha256>.json   — one file per request, the cache itself
    outputs/generations/<stage>.jsonl     — append-only log of EVERY generation
                                            (CLAUDE.md: never let a generation exist only in
                                            memory or scrollback)

Invariants
    - Key = sha256 of a CANONICAL json dump of {model, messages, temperature, top_p, max_tokens,
      response_format, seed, stage, item_id, sample_idx}. Sort keys — dict ordering must not
      change the key.
    - `sample_idx` IS LOAD-BEARING, not bookkeeping. Rejection sampling draws
      n_samples_per_vignette (default 4) at temperature 1.0, and every one of those draws shares
      the model, messages, temperature and item_id. Without sample_idx in the key, draws 2-4 are
      cache hits on draw 1, the judge gets one identical trace to choose between, and the whole
      point of sampling above greedy is silently lost.
    - Writes are ATOMIC: write to <name>.tmp then os.replace. A pod preempted mid-write must not
      leave a truncated json that poisons the next run's cache hit.
    - The cache is keyed on request content, so changing a prompt template correctly invalidates
      only the affected entries.

TODO
    - cache_key(request: dict) -> str
    - get(stage, key) -> dict | None  /  put(stage, key, response) -> None
    - log_generation(stage, record) — append to the jsonl, always, hit or miss
    - A `--no-cache` escape hatch for deliberate regeneration.
    - Decide whether to record the cost of a cache HIT as zero (it is) in cost.json, so a
      resumed run's reported spend is the real incremental spend.
"""

from __future__ import annotations


def cache_key(request: dict) -> str:
    raise NotImplementedError


def get(stage: str, key: str) -> dict | None:
    raise NotImplementedError


def put(stage: str, key: str, response: dict) -> None:
    raise NotImplementedError


def log_generation(stage: str, record: dict) -> None:
    raise NotImplementedError
