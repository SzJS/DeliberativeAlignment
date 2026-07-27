"""STUB — inspect task: YES/NO decision accuracy on the frozen test vignettes.

    inspect eval evals/rrc_decision.py@rrc_decision --model vllm/<path> -T arm=sdf_sft

This is the direct successor to dr_rrc_replication/src/evaluate.py.

NOT THE HEADLINE METRIC. The vignette dataset is separable by difficulty — all hard cases are
ground-truth YES, all easy cases NO — so difficulty alone predicts the label, and decision
accuracy therefore cannot show whether deliberation CONTENT helps. rrc_procedure is the task
that can. Report this one, but do not lead with it.

TODO
    @task
    def rrc_decision(arm: str = "sdf_sft"):
        return Task(
            dataset=rrc_dataset(cfg),
            solver=rrc_solver(spec=spec_for_arm(arm, cfg), few_shot=needs_few_shot(arm)),
            scorer=[decision_accuracy(), format_adherence()],
        )

    - `arm` is a TASK ARGUMENT, not a separate task — `base` and `spec_in_context` share a model,
      so they cannot be distinguished by model string alone. See eval_lib/arms.py.
    - Load cfg via config_utils.load_config(snapshot=False): inspect may run this file many times
      in one eval_set, and each snapshot write would race the others.
    - CWD: the package import resolves from anywhere, but `load_config("config.yaml")` and
      `paths.artifacts` do NOT — they are cwd-relative, and inspect loads this file by path from
      whatever directory you invoked it in. Either resolve the config path relative to
      `__file__`, or take it as a task arg (`-T config=/abs/path`). The usage line above assumes
      cwd is dr_rrc_sdf/.
"""

from __future__ import annotations
