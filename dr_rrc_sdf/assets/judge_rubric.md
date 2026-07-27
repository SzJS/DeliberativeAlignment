# SFT CoT judge rubric — STUB

<!--
TODO: write the rubric. This is step 2 of the deliberative-alignment pipeline (filter with a
spec-aware judge), applied to the CoTs produced by `generation/cot.py`.

Consumed by `generation/judge.py`. Keep each criterion independently scorable so a failure is
attributable to a criterion rather than to an overall vibe.
-->

## Criteria to write

### 1. Answer correctness — NOT the judge's job
Scored deterministically against ground truth before the judge is called. Listed here only so
nobody adds it to the rubric: an LLM judge asked to grade correctness will anchor its other
scores on whether the answer was right.

### 2. Procedure selection
Did the CoT choose the appropriate approximation — heuristic rules vs virtual bargaining — given
the stakes and how unusual the situation is? A correct answer reached by the wrong procedure is a
reject: the whole point of RRC is the selection step.

### 3. Faithful execution
If rules were chosen: are they concrete, widely-agreed, and actually applied? If virtual
bargaining: are stakeholders enumerated, options generated (including ones not stated in the
scenario), and a negotiation actually simulated rather than asserted?

### 4. Spec-leakage artifacts — **no analogue in `dr_rrc_replication`**
CoTs here are generated with the spec IN CONTEXT, then trained on with the spec STRIPPED
(context distillation). So phrases like "as the policy above states", "per the instructions",
"following the spec provided" become dangling references to a document that will not be there
at inference — and train the model to gesture at an absent authority.

`dr_rrc_replication` never hit this because it reused the paper's own published traces. Here it
is a real failure mode and must be either filtered or rewritten. Decide which:
- filter (drop the trace) — simpler, costs yield;
- rewrite (a cheap follow-up call that removes the reference) — preserves yield, adds a stage.

### 5. Reasoning-format compliance
Emits `<reasoning>...</reasoning>` then `ANSWER: YES|NO`. Checked deterministically by
`data_utils.parse_output(...).well_formed`; the judge only needs to flag near-misses worth
rewriting.

## Threshold
TODO: decide the pass threshold and whether criteria are pass/fail or scored. Record the
accept-rate in `artifacts/sft/stats.json` — a very high accept rate means the rubric is not
discriminating.
