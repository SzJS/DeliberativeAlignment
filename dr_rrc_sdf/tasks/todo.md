# dr_rrc_sdf — build todo

Plan: `/home/jazon/.claude/plans/hi-claude-i-want-partitioned-sundae.md`
Design: `experiment_description.md` · Checks: `VERIFICATION.md`

## Scaffold (done)

- [x] Directory tree, `.gitignore` (note: `uv.lock` deliberately NOT ignored here)
- [x] `pyproject.toml` — vLLM-anchored pin set, hatchling src-layout package, plus a
      mutually-exclusive `gpu-unsloth` extra for a separate `.venv-unsloth` training environment
      (declared via `[tool.uv] conflicts`)
- [x] `config.yaml` — sectioned per stage, dotted-path smoke overrides
- [x] `assets/` — vendored `granite_chat_template.jinja`; `rrc_spec.md`, `judge_rubric.md`,
      `sdf_manifest.json` as placeholders
- [x] Ported working code: `config_utils.py` (nested overrides), `prompts.py`, `data_utils.py`
- [x] Stubs: `generation/` (16), `training/` (7), `eval_lib/` (5 + `__init__`), `evals/`
      (3 tasks + `run_evals.py` driver)
- [x] `scripts/` — `run_all.sh`, `lib.sh`, `preflight.sh`, `fetch_chat_template.py`
- [x] Docs — `README.md`, `VERIFICATION.md`, `experiment_description.md`, this file
- [x] Verified: `RRC_PROMPT` byte-identical to experiment 1 (`sha b50a1b7b95ec`), every `.py`
      parses, shell scripts pass `bash -n`, `config.yaml` parses, no R1 machinery leaked in

## Blocking decisions (nothing downstream can run until these land)

- [ ] **Write `assets/rrc_spec.md`.** Blocks all generation. Decide: superset containing
      `RRC_PROMPT` verbatim (recommended, and what `generation/spec.py` is scaffolded to assert)
      vs `RRC_PROMPT` alone vs a different document with the confound accepted and reported.
- [ ] **Set `OPENROUTER_API_KEY`** in the repo-root `.env` — currently a 0-byte file.
- [ ] **Pick generation models** per stage in `generation.models` (cheap for fan-out, strong for
      document writing and judging), then price a run with `--dry-run`.
- [ ] **Pick the public instruction dataset** for the SFT mix — licence + size — and the
      RRC:instruction ratio. Record the licence in README's data-provenance section.
- [ ] **Pick the `sdf.replay` dataset and fraction** (generic raw text, distinct from the
      instruction mix).

## Implementation, in dependency order

- [ ] `generation/`: `openrouter.py` → `cache.py` → `jobs.py` → `schemas.py` (the shared spine;
      everything else is one `run_job` call)
- [ ] `generation/spec.py` with the `RRC_PROMPT in spec` assertion
- [ ] `generation/sdf.py` + `sdf_pipeline.py` — the five MSM stages, `--dry-run`, `--emit-manifest`
- [ ] `generation/vignettes_paper.py` — a ~60-line REDUCTION of experiment 1's
      `datasets_common.py`, not a copy (we need story + label + scenario key; the trace-mining
      machinery is dead weight here)
- [ ] `generation/splits.py` — **port `make_splits` verbatim** from
      `dr_rrc_replication/src/prepare_data.py`, keeping all three overlap assertions
- [ ] `generation/cot.py`, `judge.py`, `instruction_mix.py`, `sft_pipeline.py`
- [ ] `training/`: `modeling.py` → `chat_format.py` → `sdf_dataset.py` → `train_sdf.py` →
      `merge.py` → `sft_dataset.py` → `train_sft.py`
- [ ] `eval_lib/` + `evals/` — arms, datasets, solvers, scorers, report, four task files
- [ ] `generation/vignettes_generated.py` — **deferred**: the ground-truth-label question is
      unresolved (see the module docstring), which is why `vignette_source: paper` is the default

## Environment (needs a GPU host — nothing has been resolved)

- [ ] `uv sync --extra gpu --extra vllm` on RunPod. **Commit the resulting `uv.lock`.**
- [ ] If using unsloth: `UV_PROJECT_ENVIRONMENT=.venv-unsloth uv sync --extra gpu-unsloth`, then
      confirm `[tool.uv] conflicts` actually lets uv resolve both stacks from one lockfile, and
      that a checkpoint written in one venv loads in the other (both pin torch 2.11 for this)
- [ ] Confirm `granite-4.1-8b-base` loads with `attn_implementation="kernels-community/flash-attn2"`,
      falling back to `sdpa` cleanly
- [ ] Sweep the SFT learning rate — `logits_scaling: 16.0` means experiment 1's `1e-4` does not
      transfer
- [ ] Smoke run end to end with `smoke.enabled: true` (3B), then clear `artifacts/`

## Known facts worth not rediscovering

- `granite-4.1-8b-base` is a **dense** `GraniteForCausalLM`; only the 30B in the 4.1 line is
  hybrid Mamba-2, so no `mamba-ssm`/`causal-conv1d`.
- The **base** tokenizer already contains `<|start_of_role|>` (100264), `<|end_of_role|>`
  (100265), `<|end_of_text|>` (100257) and a real `<|pad|>` (100256). So the instruct chat
  template installs with **no added tokens and no embedding resize** — which matters because
  `tie_word_embeddings: true` makes a resize destructive.
- `pad != eos` here. Experiment 1's `pad_token = eos_token` is wrong for this model and would
  make padding indistinguishable from a document boundary in the packed SDF stage.
- `logits_scaling: 16.0` changes loss magnitude and effective LR relative to Llama-shaped models.
- Granite is Llama-shaped otherwise, so the LoRA target-module list ports unchanged. **Never** add
  `lm_head` or `embed_tokens`.
- `unsloth` 2026.7.5 requires `transformers<=5.5.0`, `trl<=0.24.0`, `torch<2.12`, `datasets<4.4` —
  mutually exclusive with vLLM 0.26 (`transformers>=5.5.3`). Hence dropped.
- `scikit-learn` is in experiment 1's deps but imported nowhere; not carried over.
- The paper's vignettes: agent 40 scenarios × 3 wording variants, development 13 × 5; agent easy
  and hard **share titles** (real leakage risk); all hard are YES, all easy NO.

## Review

- [x] Plan reviewed by the `Plan` subagent before implementation; its findings on the import root,
      pad token, tied embeddings, merge step, and spec-identity confound were folded in
- [ ] Spawn `code-reviewer` on the scaffold
