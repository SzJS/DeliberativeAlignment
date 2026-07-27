"""STUB — load the RRC spec that seeds SDF generation, and enforce the one-spec invariant.

Contract
    `load_spec(cfg)` reads assets/rrc_spec.md and returns its text plus sha256.

Invariants — the reason this module exists at all
    - `prompts.RRC_PROMPT` MUST appear VERBATIM in the loaded spec. The SDF corpus is generated
      *about* this document; the `spec_in_context` eval arm feeds a spec in-context; the SFT CoT
      generator prompts with a spec. If those are different documents, the model was taught
      spec A and evaluated against spec B — a straight confound in the headline comparison.
      The assert is a one-line tripwire that makes the design self-checking.
    - Editing assets/rrc_spec.md INVALIDATES an already-generated SDF corpus. The sha256 goes
      into every artifact manifest so the mismatch surfaces as an error rather than as a
      mysteriously flat result.

TODO
    - load_spec(cfg) -> (text, sha256); raise if the file is still the placeholder.
    - assert prompts.RRC_PROMPT in text, with an error message that explains the invariant
      rather than just failing.
    - Snapshot to artifacts/spec/{spec_snapshot.md, spec.sha256} on every run.
    - Decide (open question in assets/rrc_spec.md) whether spec_in_context feeds the FULL
      rrc_spec.md or only the RRC_PROMPT core. Whichever — it must be the same text the corpus
      was generated from.
"""

from __future__ import annotations


def load_spec(cfg: dict) -> tuple[str, str]:
    raise NotImplementedError
