"""STUB — inspect solvers. One prompt constructor shared with training.

Contract
    rrc_solver(spec) = chain(system_message(...), prompt_template(...), generate())

Invariant — the reason this module is thin
    Build messages with data_utils.build_messages(story, spec=spec) — the SAME function
    sft_dataset.py uses. dr_rrc_replication had separate builders for the SFT input and the eval
    baselines; two constructors of near-identical message lists can drift, and a drift between
    the training input and the eval input silently destroys the context-distillation comparison
    while both sides still "work". Do not add an eval-only variant here.

TODO
    - rrc_solver(spec: str | None, few_shot: bool = False)
    - The few_shot exemplars for untrained arms: FORMAT ONLY. Demonstrate <reasoning> tags and
      the ANSWER: line on a trivial non-moral example. If the exemplars contain RRC reasoning,
      they become the intervention and the `spec_in_context` arm stops being a control.
    - max_connections from cfg so the eval is concurrent — inspect handles this natively, which
      is the point of replacing dr_rrc_replication's synchronous `for ex in tqdm(test)` loop.
"""

from __future__ import annotations


def rrc_solver(spec: str | None = None, few_shot: bool = False):
    raise NotImplementedError
