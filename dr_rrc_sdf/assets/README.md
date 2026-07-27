# `assets/` — committed reproducibility anchors

`data/` and `artifacts/` are git-ignored, so anything that must survive a fresh clone and
reproduce a run byte-for-byte lives here. Two kinds of file:

**Vendored upstream artifacts** — copies of external files we depend on, committed so a silent
upstream edit cannot move our numbers.

- `granite_chat_template.jinja` — copied from `ibm-granite/granite-4.1-8b` (the *instruct*
  sibling). `granite-4.1-8b-base` ships no chat template, so we install this one onto the base
  tokenizer at load time. Safe because the **base tokenizer already contains** every marker the
  template emits — `<|start_of_role|>` (100264), `<|end_of_role|>` (100265), `<|end_of_text|>`
  — so no tokens are added and **no embedding resize is needed**. That matters here: the model
  has `tie_word_embeddings: true`, so a resize would rebuild tied input+output embeddings and
  train rows that start as noise. Refresh with `scripts/fetch_chat_template.py`.

**Generated-once pipeline inputs** — LLM-produced artifacts that are expensive to regenerate and
whose exact content defines a run. (Same role `framed_deliberation.json` plays in
`dr_rrc_replication`.)

- `rrc_spec.md` — the RRC spec that seeds SDF document generation. **Placeholder; see the TODO
  inside.**
- `sdf_manifest.json` — output of SDF generation stages 1–4 (domains → document types →
  character assertions → document ideas). Cheap to generate, defines the whole corpus, so it is
  committed while the ~GB of stage-5 documents is not. **Placeholder.**
- `judge_rubric.md` — the grading rubric for the SFT CoT filter. **Stub.**
