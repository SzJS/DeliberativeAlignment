# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research monorepo for a **pluralistic alignment** project: teach a model to approximate
collective stakeholder deliberation in its chain-of-thought, by grafting the alignment
*target* of **Resource-Rational Contractualism** (RRC, arxiv 2506.17434) onto training
*pipelines* borrowed from **Deliberative Alignment** (arxiv 2412.16339) and **Model Spec
Midtraining** (MSM, arxiv 2605.02087). See `project_description.md` for the full thesis. The core
idea: unlike Deliberative Alignment, which aligns to static rules, RRC also gives a principled
procedure for *diverging* from those rules when stakes and disagreement are high.

The planned training pipeline (`project_description.md`, "Training pipeline") is two stages:

1. **SDF** (≈ MSM, *Teaching Claude Why*) — train on synthetic documents that discuss an RRC
   spec, as plain next-token prediction, so the model acquires the *reasons* before it sees any
   demonstrations.
2. **SFT** (≈ Deliberative Alignment) — generate RRC reasoning traces with the spec in context,
   filter them with a judge, then fine-tune with the spec **stripped** (context distillation).

Deliberative alignment's step 4 (**RL with the CoT hidden from the reward model**) is
deliberately **out of scope** for all current de-risking work, to cut cost.

## Layout — two `dr_` ("de-risking") experiments

- **`dr_rrc_replication/`** — **experiment 1, the SFT-only baseline.** Fine-tunes a small
  DeepSeek-R1-Distill model on the RRC paper's *own* reasoning traces with the spec removed from
  the prompt (context distillation), instead of prompting. Complete and runnable.
- **`dr_rrc_sdf/`** — **experiment 2, SDF → SFT on a base model.** Adds the synthetic-document
  stage before SFT, switches to `ibm-granite/granite-4.1-8b-base`, generates its own data via
  OpenRouter, and evaluates with `inspect_ai`. **Currently a scaffold**: dependencies and docs are
  real, every pipeline module is a stub, nothing has been executed.

The two are **self-contained and deliberately do not share code** — their dependency sets are
physically incompatible (see below), and experiment 1 is a finished experiment whose numbers must
stay reproducible. Shared prose lives at the repo root; tooling does not.

> ⚠️ **The invariants in each experiment's section below are scoped to that experiment.** In
> particular experiment 1's `<think>` tags, `START_OUTPUT` parsing, and unsloth pins are
> DeepSeek-R1-Distill artifacts and must not be carried into `dr_rrc_sdf`.

---

# Experiment 1 — `dr_rrc_replication`

A `uv`-managed Python package (`package = false`, run scripts with `uv run`). Deps are split so
**data-prep stages run CPU-only** and only training/eval need the GPU stack.

```bash
uv sync                              # CPU-only: data prep + eval scoring deps
uv sync --extra gpu                  # add torch/transformers/trl/peft for training
uv sync --extra gpu --extra unsloth  # optional faster/lower-VRAM LoRA
# Optional flash-attention for the transformers fallback path (unsloth has its own
# kernels; train.py auto-degrades to sdpa if this isn't installed):
#   uv pip install flash-attn --no-build-isolation
# If the (RunPod) base image ships its own CUDA torch:
#   uv venv --system-site-packages && uv sync --extra gpu

# Full pipeline (reads config.yaml)
bash scripts/run_all.sh

# Or step by step
uv run python src/download_data.py   # clone RRC_experiments repo into data/
uv run python src/inspect_data.py    # confirm workbook sheet layout
uv run python src/prepare_data.py    # -> artifacts/{train,val,test}.jsonl + splits.json
uv run python src/train.py --check   # print the loss mask and EXIT (run before training)
uv run python src/train.py           # LoRA SFT -> artifacts/adapter/
uv run python src/evaluate.py        # -> artifacts/metrics.json + eval_transcripts.jsonl

# Every script takes --config; run_all.sh accepts an alternate: bash scripts/run_all.sh other.yaml
```

There is **no test suite, linter, or CI**. Verification is manual — `VERIFICATION.md` is the
checklist of commands + expected outputs for each stage.

**Smoke test before any real run:** set `smoke: true` in `config.yaml` (swaps to the 1.5B base,
1 epoch, ~20 train / ~10 eval examples) and run the pipeline to check it executes end-to-end.
Accuracy is meaningless at smoke scale. Set `smoke: false` for the real 7B run.

Naming note: the directory and Python import root are `dr_rrc_replication`; only the pyproject
`[project].name` still reads `dr-rcc-replication` (cosmetic — `package = false`).

## Architecture of the SFT pipeline

`config.yaml` drives everything via `config_utils.load_config` (which applies `smoke_overrides`
when `smoke: true` — a **flat** `dict.update`, so overrides are top-level keys only — and autoloads
the repo-root `.env`, gitignored, into `os.environ`).

- **`datasets_common.py`** — schema knowledge for the paper's result workbooks. Knows the file
  list (`RESULT_FILES`: agent/development × easy/hard), sheet naming (`"{model} {approach}"`),
  and normalizes two slightly different column schemas into one frame. `load_condition` returns
  accuracy==1 traces per `condition`: `rrc_only`; `best_per_vignette` (prefers deliberation on
  hard cases); or `framed_deliberation` (best_per_vignette with an RRC preamble on hard cases).
- **`framed_deliberation.py`** — builds the `framed_deliberation` targets: an RRC procedure-selection
  preamble (LLM-written, committed to `assets/framed_deliberation.json`) + the paper's full
  virtual-bargaining deliberation verbatim. Rebuild the sidecar with `scripts/build_framed_sidecar.py`.
- **`prepare_data.py`** — builds leakage-safe splits and writes jsonl. **Splits on scenario key,
  never on story rows**, so all wording-variants and the easy/hard versions of a scenario stay on
  one side. Two modes: `random` (stratified, ~balanced test set) and `stakes_generalisation`
  (train on easy/low-stakes, test on hard/high-stakes — the generalisation experiment; here
  same-scenario easy-in-train/hard-in-test is intentional). The test set is built from the
  **unfiltered** held-out vignettes so accuracy and the `paper_rrc` baseline are real numbers.
- **`data_utils.py`** — the single source of truth for parsing and message construction.
  `parse_output`/`parse_answer` turn any raw text (model output OR the paper's `response_text`)
  into a YES/NO decision. Everything generated here ends in a **bare YES/NO** after `</think>`;
  the parser also recognizes the paper's `START_OUTPUT` block so its result files score identically.
- **`prompts.py`** — the RRC spec (`RRC_PROMPT`) and story template are verbatim from the paper
  (including original typos like "recommendadtion", "aproximation" — **do not fix them**).
- **`train.py`** — LoRA SFT with **manual loss masking**. Reasoning target uses the base model's
  native `<think>…</think>`; the answer is a **bare `YES`/`NO`**. Prefers `unsloth` and
  **auto-falls back to transformers+peft** if it can't import.
- **`evaluate.py`** — scores four methods on the same frozen test vignettes: `sft`, `no_thinking`
  (floor), `rrc_incontext` (prompt-vs-distill), `paper_rrc` (the paper's own answer, a free lookup).

### Invariants — experiment 1 only

- **One parser, every source.** All answer sources are scored by `parse_output`. If you change the
  generated answer format, change `prompts.py` (`FORMATTING_INSTRUCTIONS`) + `build_completion` +
  the parser together, and re-run the `--check` loss-mask step.
- **Context distillation depends on `strip_spec`.** With `strip_spec: true` the RRC spec must NOT
  leak into the SFT input (VERIFICATION 1.4 asserts this).
- **Never right-truncate a training example.** `build_example` returns `None` for over-length
  examples (dropped and counted) rather than truncating, which would silently cut the answer + eos
  and train an answer-less, non-terminating target. If p95 > `max_seq_len`, raise `max_seq_len`.
- **Splitting is leakage-safe by construction** — `prepare_data.make_splits` hard-asserts no
  scenario-group overlap across splits. Don't weaken those assertions.

---

# Experiment 2 — `dr_rrc_sdf`

**Status: scaffold.** `pyproject.toml`, `config.yaml`, the docs, and three ported modules
(`config_utils`, `prompts`, `data_utils`) are real. Everything under `generation/`, `training/`,
`eval_lib/`, and `evals/` is a stub with a Contract/Inputs/Outputs/Invariants docstring and
`raise NotImplementedError`. **Nothing has been executed** — no training, no downloads, no API
calls, and the dependency ranges have never been resolved.

Read `dr_rrc_sdf/experiment_description.md` for the design and pre-registered risks,
`VERIFICATION.md` for the (written, unexecuted) checklist, and `tasks/todo.md` for what's next.

```bash
uv sync                                 # CPU-only: generation pipelines + eval authoring
uv sync --extra gpu                     # + training
uv sync --extra gpu --extra vllm        # + in-process vLLM serving for eval

# Optional SECOND venv for unsloth — it CANNOT share an environment with vLLM:
UV_PROJECT_ENVIRONMENT=.venv-unsloth uv sync --extra gpu-unsloth

bash scripts/preflight.sh               # secrets, spec, GPU, disk. No API calls.
bash scripts/run_all.sh                 # the sdf_sft arm
bash scripts/run_all.sh --arm sft_only  # the ablation arm (SDF stages auto-skipped)
bash scripts/run_all.sh --dry-run       # print the plan, run nothing
```

**Seven stages** (`sdf_gen → sdf_train → merge_sdf → sft_gen → sft_train → merge_sft → eval`),
each independently skippable and resumable at two levels: coarse stamps in `artifacts/.stamps/`,
plus each stage's own mechanism (response cache / `resume_from_checkpoint` / inspect
`eval_set(log_dir=)`).

**Comparison arms:** `sdf_sft` (treatment), `sft_only` (the ablation isolating SDF), `sdf_only`
(SFT's own init checkpoint, so it is free — and it is what tells you whether SDF installed
anything at all). `base` and `spec_in_context` exist but are off by default.

### Invariants — experiment 2 only

- **The two training stages are deliberately asymmetric, and the asymmetry IS the experiment.**
  SDF is plain next-token prediction — no chat template, no loss mask, `labels == input_ids`.
  SFT is chat-formatted with completion-only loss. Each trainer's `--check` asserts the *inverse*
  of the other's; run both. If either stage acquires the other's mechanics, it is wrong.
- **One spec, three consumers, byte-identical.** `prompts.RRC_PROMPT` feeds the `spec_in_context`
  arm, the SFT CoT generation prompt, and (as a verbatim substring) `assets/rrc_spec.md`, the SDF
  corpus seed. Any edit makes the arms non-comparable; an edit after generation invalidates the
  corpus. `spec_sha256()` goes into every artifact manifest.
- **Merge is load-bearing, not a convenience.** SFT must initialise from *merged* SDF weights.
  Stacking a second LoRA on an unmerged first adapter silently trains against the base model —
  a clean run and a null SDF result that is actually a plumbing bug, with no error message.
- **Granite specifics:** `tie_word_embeddings: true` → never put `lm_head`/`embed_tokens` in LoRA
  targets, and never `resize_token_embeddings`. `pad` (`<|pad|>`, 100256) ≠ `eos`
  (`<|end_of_text|>`, 100257) — do **not** copy experiment 1's `pad_token = eos_token`.
  `logits_scaling: 16.0` changes loss magnitude and effective LR, so experiment 1's `1e-4` does
  not transfer. The LoRA target-module list *does* port unchanged (Granite is Llama-shaped).
- **The chat template is vendored**, not fetched at runtime. The base model ships none; the
  instruct sibling's is committed to `assets/`. Safe because every marker it emits is already in
  the base vocab (verified) — so no added tokens, no resize.
- **Config overrides are dotted paths that must already exist.** `set_by_path` raises on a typo,
  where experiment 1's flat `cfg.update()` would silently add a dead key.
- **No `PYTHONPATH` shim.** This experiment installs as a real editable package (src-layout,
  hatchling), because `inspect eval` loads task files by path in its own process and would never
  see a PYTHONPATH set by a shell script. If imports fail, `uv sync` is the fix.
- **The dependency stacks are mutually exclusive by construction.** `unsloth` caps
  `transformers<=5.5.0`; `vllm` requires `>=5.5.3`. They live in separate venvs, declared via
  `[tool.uv] conflicts`, and both pin `torch 2.11` so checkpoints are interchangeable. unsloth is
  **LoRA-only** and does not accelerate the full-FT SDF default.
- **`rrc_decision` accuracy is not the headline metric** — see the dataset caveat below.

---

## Data provenance

- **RRC vignettes and traces**: <https://github.com/mint-philosophy/RRC_experiments> (MIT), cloned
  into each experiment's own `data/` by its own downloader. Experiment 1 mines the paper's
  *traces*; experiment 2 uses only the vignettes and generates its own traces.
- **Base models**: experiment 1 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (R1 traces go with the
  grain, since the base was distilled from R1); experiment 2 `ibm-granite/granite-4.1-8b-base`
  (Apache 2.0), with `granite-4.1-3b-base` for smoke runs.
- **Generation** (experiment 2): OpenRouter via the async `openai` SDK, `OPENROUTER_API_KEY` from
  the gitignored repo-root `.env`.
- `data/`, `artifacts/`, and `outputs/` are git-ignored. `assets/` holds committed reproducibility
  anchors precisely because of that.

**Dataset caveat that applies to BOTH experiments:** the paper's vignettes are separable by
difficulty — **all hard cases are ground-truth YES, all easy NO**. Difficulty alone predicts the
label, so YES/NO accuracy can never show whether deliberation *content* helps; judge the CoT
structure too. Under `stakes_generalisation` the hard test set is all-YES (majority baseline 1.0)
— read it with the easy `val` (all-NO) to catch a YES-collapse.

## Project conventions

Follow `~/Documents/CLAUDE.md` (plan-first, simplicity, minimal-impact, root-cause fixes). Repo
specifics: write plans to the relevant experiment's `tasks/todo.md` and corrections to
`tasks/lessons.md`; "proving it works" means running the relevant `VERIFICATION.md` stage.

## Compute & training best practices

Apply these whenever writing or running training, eval, or data-generation code. Current work is
single-GPU LoRA SFT, but **write code that scales to the multi-GPU / RL work** without a rewrite.

**Parallelize as much as possible.**
- Make LLM and I/O calls **async and concurrent** — data generation, judge filtering, and eval
  should fire batched/concurrent requests, never a synchronous loop. Bound concurrency and back
  off on rate limits. (`inspect_ai` gives this for free on the eval side; `generation/jobs.py` is
  the equivalent for generation.)
- **Data parallelism** (DDP / `accelerate` / `torchrun`) to shard the batch across GPUs, and
  **tensor parallelism** to split large models that don't fit on one card. Full-parameter 8B
  training needs FSDP.
- Overlap data loading with compute (`num_workers`, prefetch, sequence packing) so the GPU never
  stalls waiting on the input pipeline.

**Use every memory/throughput lever.**
- **Flash attention** for fast, memory-efficient attention. Experiment 2 prefers
  `attn_implementation="kernels-community/flash-attn2"` (loads from the Hub via `kernels`) over
  compiling a `flash-attn` wheel against a fresh torch.
- **Quantization** — 4-bit/8-bit QLoRA (`load_in_4bit`) when VRAM-bound; bf16 otherwise.
- **LoRA/PEFT** over full fine-tuning, *except* where the method demands otherwise: experiment
  2's SDF stage defaults to full-parameter FT, because LoRA may lack the capacity to install
  knowledge rather than style.
- Drive **GPUs to ~100% utilization** and **RAM/VRAM near full** — raise batch size / grad-accum /
  packing until the run is *compute*-bound, not idle. Profile rather than guessing.

**Break gracefully — make everything resumable.**
- Checkpoint frequently and make every long job **resume from the last checkpoint**; cache
  intermediate artifacts so a crash at hour 6 costs minutes, not the run. Assume RunPod spot pods
  get preempted.
- Prefer idempotent, re-runnable stages (committed data sidecars, content-addressed response
  caches, stage stamps) over one-shot state.

**Observe everything.**
- Log **all** runs to **Weights & Biases** — loss, LR, grad norm, throughput, GPU/mem utilization,
  eval metrics, sample generations, and the full config. No un-tracked runs. The API key comes
  from `WANDB_API_KEY` in the gitignored `.env`; the default project is `deliberative-alignment`.
- **Save all model outputs.** Whenever a model generates text (data generation, judge filtering,
  eval transcripts, sample generations), persist it under `outputs/` — never let a generation
  exist only in memory or scrollback. `outputs/` is git-ignored.
- **Save all terminal outputs.** Capture every command's output to a file under `outputs/` while
  still printing it, e.g. `uv run python src/train.py 2>&1 | tee outputs/train.log`. Experiment
  2's `scripts/lib.sh` does this automatically per stage.
