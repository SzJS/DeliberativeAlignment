"""STUB — async OpenRouter client.

Contract
    One shared `AsyncOpenAI` pointed at OpenRouter, with our own retry/backoff and a hard cost
    ceiling. Every LLM call in this experiment goes through `complete()`.

Inputs
    cfg["generation"]: base_url, api_key_env, max_concurrency, max_retries, timeout_s,
    max_cost_usd, models.{decompose,assertions,ideas,writer,vignette,cot,judge}

Outputs
    Completion text (or a validated object when a schema is supplied), plus accumulated usage
    written to artifacts/<stage>/cost.json.

Invariants
    - The SDK's own retry is DISABLED (`max_retries=0`); we retry with tenacity so backoff is
      observable, loggable, and countable. Silent SDK retries make rate-limit tuning guesswork.
    - `max_cost_usd` is a HARD STOP, not a warning. A generation bug must not be able to burn
      the budget while unattended.
    - Never called in a synchronous loop — see jobs.run_job. (CLAUDE.md: LLM calls are async and
      concurrent, bounded, with backoff.)

TODO
    - build_client(cfg) -> AsyncOpenAI(base_url=..., api_key=os.environ[cfg api_key_env],
      max_retries=0, timeout=timeout_s); set default_headers HTTP-Referer / X-Title so usage is
      attributable in the OpenRouter dashboard.
    - Raise a clear, actionable error if the api_key_env var is unset. NOTE: the repo-root .env
      is currently EMPTY — this will be the first thing anyone hits. scripts/preflight.sh checks
      it before any stage runs.
    - complete(client, model, messages, *, temperature, max_tokens, response_format=None) with
      tenacity: wait_random_exponential(multiplier=1, max=60), stop_after_attempt(max_retries),
      retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError,
      InternalServerError)). Do NOT retry on BadRequestError — that is a bug in our prompt.
    - CostTracker: accumulate `usage` from each response; expose spent_usd(); raise BudgetExceeded
      once it crosses max_cost_usd.
    - estimate_cost(prompts, model) for the --dry-run path in jobs.py: render prompts, count
      tokens, price them, make ZERO API calls.
"""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Raised when accumulated spend crosses generation.max_cost_usd."""


def build_client(cfg: dict):
    raise NotImplementedError


async def complete(client, model: str, messages: list[dict], **kwargs) -> str:
    raise NotImplementedError


def estimate_cost(prompts: list[list[dict]], model: str) -> dict:
    raise NotImplementedError
