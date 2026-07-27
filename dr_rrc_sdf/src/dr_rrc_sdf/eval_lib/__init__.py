"""Reusable inspect_ai components: arms, datasets, solvers, scorers, reporting.

The `@task` definitions themselves live in the top-level `evals/` directory, because inspect
loads task files BY PATH (`inspect eval evals/rrc_decision.py@rrc_decision`) in its own process.
Keeping reusable code here and only task definitions there is the idiomatic split — and it works
because dr_rrc_sdf is a real installed package, so `evals/` can import from it regardless of cwd.
"""
