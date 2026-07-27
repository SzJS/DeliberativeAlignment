"""STUB — scorers. Prefer inspect's built-ins; hand-rolled parsing stays out of the scoring path.

The answer format was chosen to make this possible: `ANSWER: YES` is what
pattern(r"ANSWER:\\s*(YES|NO)") + match() already understand, so decision scoring needs no custom
parser at all. That is the main practical win of moving off dr_rrc_replication's evaluate.py.

TODO — four scorers:

  decision_accuracy()
      A built-in extraction scorer — `pattern(r"ANSWER:\\s*(YES|NO)")` — not a custom parser.
      CAVEAT: inspect's pattern() takes the FIRST match; data_utils.parse_answer takes the LAST.
      For a well-formed output there is exactly one, so they agree. For a malformed one they can
      disagree, which would let decision_accuracy and format_adherence describe the same output
      differently. Either anchor the pattern to the end of the output, or score both from
      parse_output and keep pattern() only as documentation of the format.
      metrics=[accuracy(), stderr()] plus grouped(accuracy(), "difficulty") and
      grouped(accuracy(), "domain").
      STDERR IS FREE HERE AND MUST BE REPORTED. dr_rrc_replication's metrics.json reports bare
      accuracies on n~47, where the difference between two arms is often inside the noise.
      Also report the majority-class baseline per slice: under split_mode:
      stakes_generalisation the hard test set is all-YES, so the majority baseline is 1.0 and a
      raw accuracy of 1.0 means nothing.

  format_adherence()
      Custom @scorer, deterministic, wrapping data_utils.parse_output(...).well_formed.
      Report alongside accuracy for EVERY arm — it is what distinguishes a reasoning failure
      from a formatting failure, especially for the untrained arms (see arms.py).

  procedure_fidelity()
      model_graded_qa() with a custom template + cfg.eval.grader_model (an openrouter/ model).
      Did the CoT (a) SELECT the right approximation — rules vs virtual bargaining — given stakes
      and unusualness, and (b) actually EXECUTE it (stakeholders enumerated, options generated,
      negotiation simulated)?
      This is the scorer dr_rrc_replication structurally could not have: its dataset is separable
      by difficulty (all hard -> YES, all easy -> NO), so decision accuracy can never show that
      deliberation CONTENT helps. This can.

  spec_citation()
      Model-graded: did the CoT invoke the correct spec clause? Pairs with the spec_recall task
      to measure what SDF installed.

Multiple scorers per task is native to inspect — no plumbing needed.

DELETED relative to dr_rrc_replication: the `paper_rrc` "free lookup" arm. This experiment
generates its own CoTs, so there is no paper answer to score, and under
vignette_source: "generated" there is no paper at all. If the reference ceiling is wanted back
under vignette_source: "paper", it belongs in report.py as a COLUMN computed from test.jsonl
metadata — not as a generative arm.
"""

from __future__ import annotations


def decision_accuracy():
    raise NotImplementedError


def format_adherence():
    raise NotImplementedError


def procedure_fidelity(grader_model: str):
    raise NotImplementedError


def spec_citation(grader_model: str):
    raise NotImplementedError
