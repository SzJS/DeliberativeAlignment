"""STUB — vignettes we generate ourselves (vignette_source: "generated").

Contract
    Produce moral-dilemma vignettes matched to the paper's structure — an easy/low-stakes tier
    and a hard/high-stakes tier across the agent and development domains — with ground-truth
    labels and leakage-safe scenario keys.

Why this source exists
    The paper's set is ~53 scenarios. That is thin for SFT, and it carries a structural defect
    (all hard -> YES, all easy -> NO) that makes decision accuracy uninformative about
    deliberation quality. Generating our own is the route to a set that breaks that confound.

OPEN QUESTION — this is the hard part, do not paper over it
    Where does the ground-truth label come from? The paper's labels are expert contractualist
    judgments. Options, none free:
      (a) LLM-generated label with the spec in context — circular: we would be measuring
          agreement with the generator, not moral correctness.
      (b) Generate the scenario FROM a target label (construct a case where the answer is X) —
          controllable, but risks generating cases that telegraph their label.
      (c) Human labelling of a small set — the only real ground truth, and expensive.
    Until this is resolved, `vignette_source: "paper"` is the default and this path is
    scaffolding only.

BREAK THE DIFFICULTY/LABEL CONFOUND
    Whatever is chosen, this generator is the one place that CAN produce hard vignettes labelled
    NO and easy vignettes labelled YES. If it reproduces the paper's confound, generating our
    own vignettes buys nothing over using theirs.

TODO
    - generate(cfg) -> list[Vignette], via jobs.run_job.
    - Assign scenario_key at generation time (a scenario id shared by its wording variants), so
      splits.py's leakage guarantees hold for this source too.
    - Report the difficulty x label contingency table in stats.json. If it is diagonal, say so
      loudly — that is the confound reappearing.
"""

from __future__ import annotations


async def generate(cfg: dict) -> list[dict]:
    raise NotImplementedError
