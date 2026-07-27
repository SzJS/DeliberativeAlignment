"""STUB — the resumable, bounded-concurrency map that every generation stage is built from.

Contract
    `run_job(items, fn, ...)` runs `fn` over `items` concurrently, caching each result, isolating
    per-item failures, and reporting progress. Every stage in sdf.py, cot.py and judge.py is one
    call to this.

Invariants
    - asyncio.Semaphore(max_concurrency) + asyncio.TaskGroup. Never a synchronous for-loop over
      API calls (CLAUDE.md), and never unbounded fan-out (instant rate-limit wall).
    - One item's failure must not kill the job. After max_retries it goes to
      artifacts/<stage>/failed.jsonl with its error, and the job continues.
    - Idempotent: re-running with the same inputs is a no-op that costs nothing (see cache.py).

TODO
    - async run_job(items, fn, *, stage, concurrency, cache, on_result=None) -> list[Result]
      where Result carries (item, value | None, error | None, cached: bool).
    - --dry-run: render and print N prompts plus an estimated token count and cost, and make
      ZERO API calls. This is the only way to size a run's cost before committing to it, and the
      only way to inspect the pipeline on a machine that shouldn't be calling APIs at all.
    - Progress via tqdm, and a periodic spend line so a long run's cost is visible as it accrues.
    - Structured-output repair loop: when a response fails schema validation, retry with the
      validation error appended to the prompt, up to N times, then quarantine to failed.jsonl.
      (Shared by every stage that wants json — keep it here, not duplicated per stage.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Result:
    item: Any
    value: Any | None
    error: str | None
    cached: bool


async def run_job(items: list, fn: Callable, *, stage: str, concurrency: int, **kwargs) -> list[Result]:
    raise NotImplementedError
