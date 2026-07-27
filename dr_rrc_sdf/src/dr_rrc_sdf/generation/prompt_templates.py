"""STUB — the prompt templates for every generation stage, in one file.

Contract
    Pure string templates plus render functions. No API calls, no I/O — so `--dry-run` can render
    and price the whole run without touching the network, and so a prompt change is a reviewable
    diff in one place.

TODO — templates for:

  SDF stages 1-5 (MSM's hierarchy). Generation constraints from project_description.md that must
  be encoded in the WRITER prompt and re-checked in the filter:
    - documents must be about the SPECIFIC NAMED MODEL, not "an AI" in general;
    - must NOT fabricate real-world details — no invented dates, authors, citations, URLs,
      institutions. (This is the constraint most likely to be violated and least likely to be
      noticed, because fabricated citations read as high-quality prose.)
    - must NOT invent values or motivations absent from the spec. The corpus is supposed to
      teach the spec, not an LLM's guess at what a contractualist assistant would also believe.
    - stage 4 ideas must each frame the content from a DIFFERENT perspective (researcher's
      internal report, user's blog post about an interaction, case study of an exchange) —
      this is the diversity mechanism; without it the corpus collapses into paraphrase.

  SFT CoT generation: spec in context + story + FORMATTING_INSTRUCTIONS -> reasoning + answer.
    Build the messages with data_utils.build_messages(story, spec=...) — do NOT hand-roll a
    second message constructor here, or the generation prompt and the eval prompt will drift.

  Judge: assets/judge_rubric.md + the vignette + the CoT -> per-criterion verdict.

  Vignette generation (sft_data.vignette_source == "generated"): scenarios matched to the
    paper's easy/hard structure. See the open question in vignettes_generated.py about labels.
"""

from __future__ import annotations
