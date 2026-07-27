"""STUB — dataclasses + JSON schemas for every generation-stage artifact.

Contract
    One place defining the shape of what each stage emits, used both to request structured
    output (`response_format` json_schema where the model supports it) and to validate what came
    back before it is written to disk.

Invariants
    - Validate BEFORE writing. An unvalidated artifact propagates a malformed record into
      training data, where it is much more expensive to find.
    - Validation failure is repairable, not fatal: retry with the error appended to the prompt
      (see jobs.py), then quarantine.

TODO — schemas for:
    SDF stages   Domain{name, description, subdomains[]}
                 Subdomain{name, description}
                 DocumentType{name, description}          (forum thread, paper intro, internal
                                                           memo, bug report, training design doc)
                 CharacterAssertion{proposition, subdomain}
                 DocumentIdea{subdomain, doc_type, perspective, summary}
                 SyntheticDocument{subdomain, doc_type, idea_id, title, text, word_count}
    SFT stages   Vignette{scenario_key, domain, difficulty, story, label}
                 CotSample{vignette_id, sample_idx, reasoning, answer, well_formed}
                 JudgeVerdict{sample_id, procedure_ok, execution_ok, spec_leakage, score, notes}
    Manifest     SdfManifest — what gets written to assets/sdf_manifest.json (see that file)

    Every artifact record should carry the spec sha256 and the generating model, so a mixed-
    provenance corpus is detectable rather than silently averaged over.
"""

from __future__ import annotations
