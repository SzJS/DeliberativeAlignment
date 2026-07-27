"""Data generation: the SDF document corpus and the SFT reasoning traces.

Both pipelines call OpenRouter through the async `openai` SDK and share one execution model:
`openrouter.py` (client + retry) -> `cache.py` (content-addressed disk cache) ->
`jobs.py` (bounded-concurrency resumable map). Every stage is one `run_job` call.
"""
