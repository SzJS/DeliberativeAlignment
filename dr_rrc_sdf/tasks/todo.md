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

## Current task — instruction mix (MSM Table 2) + switch to an instruct model

Decided 2026-07-28. Supersedes the "Pick the public instruction dataset" blocking item above.

**Design change: SDF runs on the INSTRUCT model, not the base model.** `ibm-granite/granite-4.1-8b`
(IBM drops the `-instruct` suffix; the *base* is the suffixed one). This inverts the "Why a base
model" argument in `experiment_description.md` §3, and adopts MSM's own §4–5 rationale instead:
the instruction mix exists to repair the incoherence that midtraining induces in an Instruct model.

Architecturally the two are identical — 40 layers, hidden 4096, vocab 100352,
`tie_word_embeddings: true`, `logits_scaling: 16.0`, pad 100256 ≠ eos 100257 — so **every Granite
invariant in CLAUDE.md survives unchanged**. What changes is that the model ships its own chat
template, so the vendored one stops being load-bearing.

- [x] **`config.yaml`** — DONE. `model.base` → `ibm-granite/granite-4.1-8b`, smoke →
      `granite-4.1-3b`; `instruction_mix.{dataset,n_examples,ratio}` replaced by the Table 2
      `sources` list plus a `scale` knob and a `filter` block; added
      `generation.models.instruction_filter`; smoke overrides retargeted at the new keys
      (`scale: 0.004`, filter off — a smoke run must make no API calls).
- [ ] **Docs for the model switch.** Rewrite `experiment_description.md` §3 ("Why a base model" →
      why instruct) and its comparison-table row. Update the CLAUDE.md experiment-2 invariant
      about the vendored template, and `instruction_mix.py`'s docstring rationale (currently "we
      start from a BASE model with no instruction-following ability at all" — no longer the
      reason; the reason is now MSM's, repairing midtraining-induced incoherence).
- [ ] **Chat template.** Instruct ships one natively. Keep vendoring for reproducibility (an
      upstream edit must not silently re-render every training example), but retarget
      `fetch_chat_template.py` at the model itself and drop the "sibling" language.
      `data_utils.install_chat_template` already handles this case — it warns and overwrites when
      the tokenizer carries its own template — so no code change is needed there, only the
      comment on line ~170 that says "granite-4.1-*-base ships no chat template".
- [ ] **Generation spine** — `openrouter.py` → `cache.py` → `jobs.py`. Pulled into scope by the
      filter decision below; already the documented dependency order. Design notes from the
      2026-07-28 pass, worth not re-deriving:
      - Ask OpenRouter for the real routed cost (`extra_body={"usage": {"include": True}}`)
        rather than pricing tokens against a table that goes stale. Count responses that come
        back *unpriced* separately — otherwise an unpriced model makes spend look like zero.
      - The budget check belongs AFTER usage is added, making `max_cost_usd` a stop rather than a
        pre-authorisation: an in-flight batch may overshoot slightly, but nothing new dispatches.
        Needs an `asyncio.Lock` around the accumulator since `run_job` fans out concurrently.
      - `--dry-run` must work on a CPU-only `uv sync`, which is exactly the machine where you want
        to size a run. So the token estimate cannot use `transformers` (it is `gpu`-extra only) —
        use a character heuristic and label it as sizing, not billing.
      - tenacity's `AsyncRetrying` with `reraise=True`; retry only
        `(RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)`.
- [ ] **`generation/instruction_mix.py`** — MSM Table 2 verbatim, 10,000 samples / ~2M tokens.
      Record shape must match whatever `splits.build_sft_records` emits (`messages` + assistant
      target), since `training/sft_dataset.py` masks both with one code path. For the multi-turn
      sources (No Robots, LongAlign) the natural mapping is: `messages` = every turn up to and
      including the final user turn, `target` = the final assistant turn, so intermediate
      assistant turns are masked as prompt. Document that choice — it is a real decision, not an
      obvious one. Guarantee ≥1 sample per source whenever `scale > 0`, or smoke runs silently
      drop the small sources (LongAlign at 216) to zero and stop exercising their schema handling.
- [ ] **`preflight.sh`** — assert `HF_TOKEN` is set and probe that `GAIR/lima` actually loads, so a
      gated-repo 401 fails before stage 4 spends money.
- [ ] **Docs** — README data provenance (9 datasets + licences), VERIFICATION.md stage check.

### The mix (Appendix B.3, Table 2)

| Source | HF path | n | Licence |
|---|---|---|---|
| No Robots | `HuggingFaceH4/no_robots` | 2,779 | cc-by-nc-4.0 |
| Tulu3 IF | `allenai/tulu-3-sft-personas-instruction-following` | 1,471 | odc-by |
| NuminaMath CoT | `HuggingFaceTB/smoltalk:numina-cot-100k` | 1,063 | apache-2.0 |
| Self-Oss-Instruct | `HuggingFaceTB/smoltalk:self-oss-instruct` | 1,064 | apache-2.0 |
| Smol-constraints | `HuggingFaceTB/smoltalk:smol-constraints` | 1,055 | apache-2.0 |
| APIGen Function-Calling | `HuggingFaceTB/smoltalk:apigen-80k` | 1,054 | apache-2.0 |
| Smol-summarize | `HuggingFaceTB/smoltalk:smol-summarize` | 984 | apache-2.0 |
| LIMA | `GAIR/lima` | 314 | other (NC), **gated** |
| LongAlign | `HuggingFaceTB/smoltalk:longalign` | 216 | apache-2.0 |

"Tulu3 IF" is an interpretation: the paper says only "Tulu3 IF" sourced from the Tulu 3 SFT mix.
The standalone persona-IF dataset is the same rows as `tulu_v3.9_personas_instruction_following`
inside `allenai/tulu-3-sft-mixture`, but far cheaper to load. Record this reading in the README.

### Filter (Appendix B.3, last paragraph)

MSM filtered *every* instruction sample for spec-misalignment with Claude Sonnet 4.6, dropping
toxic data, samples where the assistant identifies as another model ("I'm GPT-4"), and
"As an AI, I have no subjective opinions/preferences". Implement via `jobs.run_job` with the
response cache, counted against `generation.max_cost_usd`. This is not optional polish: unfiltered
identity and no-preferences boilerplate directly contradicts what SDF installs.

Note this is a DIFFERENT filter from `judge.py`, which grades RRC CoTs against the rubric. It
lives in `instruction_mix.py` and shares only the `run_job` plumbing.

### Pre-registered risks

- **Dilution.** 10k instruction samples against a few hundred RRC vignettes puts the RRC signal at
  ~3% of the SFT set. MSM's smallest AFT scale was 1,250 against the same 10k, so we sit below
  their floor. Mix size must be a config knob, sweepable without a code change.
- **Table 2 was sized for repairing Instruct models**, which is now our setting too — so 2M tokens
  is the right starting point rather than a number borrowed from a different regime.
- Backfilling to hold a source's target count after filtering must draw from that same source, or
  the realised mixture silently drifts from Table 2. Record realised counts in the manifest.

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
