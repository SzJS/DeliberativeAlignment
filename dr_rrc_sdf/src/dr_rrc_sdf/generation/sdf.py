"""STUB — MSM's five-stage hierarchical synthetic-document pipeline.

Contract
    Turn one spec document into a diverse corpus of documents that discuss it, via a hierarchy
    that produces volume WITHOUT collapsing diversity (the whole reason MSM is hierarchical
    rather than "generate 5,000 documents about this spec").

Stages (project_description.md, "Model Spec Midtraining" -> data generation pipeline)
    1. stage1_domains(spec)                     spec -> coherent, non-overlapping domains
                                                (1-4 words each), collectively covering the spec.
                                                BIAS TOWARD FEWER BROAD DOMAINS over many narrow.
    2. stage2_doc_types(subdomain)              document types that would plausibly exist on the
                                                internet if the spec content were true, and would
                                                carry high-signal information about the assistant.
    3. stage3_assertions(subdomain)             explicit propositions about the assistant's
                                                values/beliefs/motivations/behaviours. Injected
                                                into downstream prompts both to keep salient
                                                facts foregrounded AND to diversify the prompt
                                                distribution — they do double duty.
    4. stage4_ideas(subdomain, doc_type)        specific ideas, each framing the content from a
                                                particular perspective.
    5. stage5_document(...)                     ONE document per (domain, subdomain, doc_type,
                                                idea), generated with the FULL SPEC in context.
                                                Corpus size is therefore the PRODUCT of the four
                                                cfg.sdf.n_* knobs — there is no separate
                                                target_docs cap, and smoke mode must shrink all
                                                four (it does).

Inputs   assets/rrc_spec.md (via spec.load_spec), cfg["sdf"], cfg["generation"]
Outputs  artifacts/sdf/01_domains.json .. 05_documents.jsonl, then corpus.jsonl (filtered),
         plus stats.json, cost.json, failed.jsonl.
         Stages 1-4 are ALSO written to assets/sdf_manifest.json and committed: they are cheap
         but they fully determine the corpus, so they are the reproducibility anchor (the role
         assets/framed_deliberation.json plays in dr_rrc_replication) while the multi-GB stage-5
         output stays git-ignored.

Invariants
    - Every stage is one jobs.run_job call — concurrent, cached, resumable.
    - Stage 5 must receive the full spec. Documents generated without it drift into generic
      "helpful AI" prose that teaches nothing specific.

TODO
    - The five stage functions above.
    - filter_corpus(): min_words; near-duplicate detection (MinHash or embedding, threshold from
      cfg.sdf.filters.near_dup_threshold); fabricated-detail check (dates, URLs, citations);
      spec-contradiction check. Report a DIVERSITY NUMBER in stats.json and look at it BEFORE
      training — diversity collapse is the main failure mode of hierarchical generation at
      de-risking scale, and it is invisible in a loss curve.
    - The replay mix (cfg.sdf.replay) is applied in training/sdf_dataset.py, not here — it is
      generic pretraining text, not part of the generated corpus.
"""

from __future__ import annotations


async def stage1_domains(spec: str, cfg: dict) -> list[dict]:
    raise NotImplementedError


async def stage2_doc_types(subdomain: dict, cfg: dict) -> list[dict]:
    raise NotImplementedError


async def stage3_assertions(subdomain: dict, spec: str, cfg: dict) -> list[dict]:
    raise NotImplementedError


async def stage4_ideas(subdomain: dict, doc_type: dict, cfg: dict) -> list[dict]:
    raise NotImplementedError


async def stage5_document(subdomain: dict, doc_type: dict, idea: dict, assertions: list[dict], spec: str, cfg: dict) -> dict:
    raise NotImplementedError


def filter_corpus(documents: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    raise NotImplementedError
