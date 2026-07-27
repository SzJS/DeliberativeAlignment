"""STUB — the comparison arms: what is being compared, and how each one is served.

An arm is a (model, solver) pair, NOT a task. A Task is the question asked of every arm; an arm
is who answers it and how. Getting this backwards produces one task per arm and makes
cross-arm comparison a post-hoc join.

Arms
    sdf_sft          artifacts/models/merged_sdf_sft    spec=None   TREATMENT: the full pipeline
    sft_only         artifacts/models/merged_sft_only   spec=None   ABLATION: no SDF stage
    sdf_only         artifacts/models/merged_sdf        spec=None   SFT's own init checkpoint, so
                                                                    this arm costs NO extra
                                                                    training. Without it an
                                                                    sdf_sft > sft_only gap cannot
                                                                    be attributed to SDF rather
                                                                    than to extra optimisation
                                                                    steps.
    base             cfg.model.base                     spec=None   floor (available, off by default)
    spec_in_context  cfg.model.base                     spec=RRC    prompt-vs-train control
                                                                    (available, off by default)

Mechanism
    `base` and `spec_in_context` SHARE a model, so an arm cannot be a bare model entry in
    eval_set's model list. Make `arm` a TASK ARGUMENT — @task def rrc_decision(arm: str = ...) —
    and have evals/run_evals.py build the explicit cross-product. The arm name then lands in the
    log's task args, so report.py groups by it for free.

FORMAT-FAIRNESS CONFOUND — design around it, do not discover it in the results
    `base` and `spec_in_context` have never been trained on our <reasoning>/ANSWER: format, so a
    low score from them may be a FORMATTING failure rather than a reasoning failure. Mitigation:
    give the untrained arms 2-3 format-only few-shot exemplars (content-neutral — demonstrating
    the format, NOT RRC reasoning, or the exemplars become the intervention), and ALWAYS report
    format_adherence next to accuracy so the confound is visible rather than buried.

TODO
    - ARMS registry: name -> {model_path_fn(cfg), spec_fn(cfg), few_shot: bool}
    - model_string(arm, cfg) -> a provider-qualified model string per cfg.eval.serve:
        vllm        -> "vllm/<path>", in-process (needs `--extra vllm` in the active venv)
        vllm_server -> an OpenAI-compatible provider pointed at cfg.eval.serve_base_url, talking
                       to a vLLM server started separately. USE THIS WHEN TRAINING IN THE
                       .venv-unsloth ENVIRONMENT: it keeps vLLM out of process entirely, which
                       is what allows unsloth (transformers<=5.5.0) and vLLM (>=5.5.3) to both
                       exist in one project. Raise if serve_base_url is null.
        hf          -> "hf/local"; additionally needs `model_path` as a model ARG
                       (`-M model_path=...`, or the keyword in Python) — it is not part of the
                       string. Slow; no extra deps.
    - Fail fast with a clear message if an arm's merged checkpoint is missing — much better than
      inspect silently evaluating the base model under the wrong arm name.
"""

from __future__ import annotations

ARMS = ("sdf_sft", "sft_only", "sdf_only", "base", "spec_in_context")


def model_string(arm: str, cfg: dict) -> str:
    raise NotImplementedError


def spec_for_arm(arm: str, cfg: dict) -> str | None:
    raise NotImplementedError
