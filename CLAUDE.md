# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research monorepo for a **pluralistic alignment** project: teach a model to approximate
collective stakeholder deliberation in its chain-of-thought, by grafting the alignment
*target* of **Resource-Rational Contractualism** (RRC, arxiv 2506.17434) onto the training
*pipeline* of **Deliberative Alignment** (arxiv 2412.16339). See `project_description.md` for
the full thesis. The core idea: unlike Deliberative Alignment, which aligns to static rules,
RRC also gives a principled procedure for *diverging* from those rules when stakes and
disagreement are high.

The four-step target pipeline (from `project_description.md`): (1) generate RRC data — rule-based
CoTs for ordinary cases, synthetic multi-stakeholder deliberation CoTs otherwise; (2) filter
with a judge; (3) SFT (process supervision) so deliberation lives in the CoT; (4) RL (outcome
supervision, CoT hidden). De-risking work deliberately **skips RL** to cut cost.

## Layout — two `dr_` ("de-risking") experiments

- **`dr_rrc_replication/`** — the only directory with runnable code. SFT-based replication of
  the RRC paper: fine-tune a small DeepSeek-R1-Distill model on the paper's own RRC reasoning
  traces with the spec removed from the prompt (context distillation), instead of prompting.
- **`dr_synthetic/`** — design handoff (`experiment_description.md`) for a *future* synthetic toy
  domain (n indivisible goods split among k stakeholders, leximin ground truth) built to make
  "when is rule-breaking correct" computable, so RCC can be measured against DA rather than judged.
  No code yet.

Naming note: the directory and Python import root are `dr_rrc_replication` (matching the paper's
**RRC** abbreviation); only the pyproject `[project].name` still reads `dr-rcc-replication`
(cosmetic — `package = false`).

## Working in `dr_rrc_replication`

This is a `uv`-managed Python package (`package = false`, run scripts with `uv run`). The
deps are split so **data-prep stages run CPU-only** and only training/eval need the GPU stack.

```bash
# Setup
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
checklist of commands + expected outputs for each stage. Use it to prove a change works.

**Smoke test before any real run:** set `smoke: true` in `config.yaml` (swaps to the 1.5B base,
1 epoch, ~20 train / ~10 eval examples) and run the pipeline to check it executes end-to-end.
Accuracy is meaningless at smoke scale — you're only checking it loads, generates, parses, and
scores without crashing. Set `smoke: false` for the real 7B run.

## Architecture of the SFT pipeline

Data flows through `src/` modules that each own one concern; `config.yaml` drives all of them
via `config_utils.load_config` (which applies `smoke_overrides` when `smoke: true`).

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
  into a YES/NO decision, so the SFT model, the in-context baseline, and the paper's files are all
  scored identically. `build_sft_messages` builds the context-distilled input (minimal system
  prompt when `strip_spec: true`); `build_completion` builds the assistant target.
- **`prompts.py`** — verbatim prompt strings from the paper (including original typos like
  "recommendadtion", "aproximation" — **do not fix them**; the in-context baseline must reproduce
  the paper). Also the tag constants and the minimal distillation system prompt.
- **`train.py`** — LoRA SFT with **manual loss masking** (labels=-100 on all prompt tokens;
  loss on the assistant target only). Reasoning target uses the base model's native
  `<think>…</think>`; the answer uses `START_OUTPUT YES/NO END_OUTPUT`. Prefers `unsloth` and
  **auto-falls back to transformers+peft** if it can't import, so a run never hard-blocks on it.
  Flash-attention (`attn_implementation`, fallback path only), W&B (`wandb_project`), and
  checkpoint resume (`resume`) are config-gated and degrade gracefully.
- **`evaluate.py`** — scores four methods on the same frozen test vignettes: `sft` (base+adapter),
  `no_thinking` (floor), `rrc_incontext` (prompt-vs-distill on the same base), `paper_rrc` (the
  paper's own answer, a free lookup). Baselines run with `model.disable_adapter()`.

### Invariants — respect these when editing

- **One parser, one format.** All three answer sources must be scored by `parse_output`. If you
  change the answer format, change `prompts.py` tag constants + `build_completion` + the parser
  together, and re-run the `--check` loss-mask step.
- **Context distillation depends on `strip_spec`.** With `strip_spec: true` the RRC spec must NOT
  leak into the SFT input (VERIFICATION 1.4 asserts this). The model is meant to recall the
  procedure from weights.
- **Never right-truncate a training example.** `build_example` returns `None` for over-length
  examples (they're dropped and counted) rather than truncating, because truncation would silently
  cut the answer + eos and train an answer-less, non-terminating target. If `prepare_data` reports
  p95 > `max_seq_len`, raise `max_seq_len` instead.
- **Splitting is leakage-safe by construction** — `prepare_data.make_splits` hard-asserts no
  scenario-group overlap across splits. Don't weaken those assertions.
- **The dataset is separable by difficulty: all hard vignettes are ground-truth YES, all easy NO.**
  So difficulty alone predicts the label — YES/NO accuracy can't show whether deliberation *content*
  helps (judge CoT structure too), and `stakes_generalisation`'s hard test set is all-YES (majority
  baseline 1.0; read it with the easy `val`, all-NO, to catch a YES-collapse).

## Data provenance

Training traces come from <https://github.com/mint-philosophy/RRC_experiments> (MIT), cloned into
`data/` by `download_data.py`. `data/` and `artifacts/` are git-ignored (regenerated by the
pipeline). Default base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. DeepSeek-R1 traces are
used by default because the base was distilled from R1, so the CoT style goes with the grain.

## Project conventions

Follow `~/Documents/CLAUDE.md` (plan-first, simplicity, minimal-impact, root-cause fixes). Repo
specifics: write plans to `dr_rrc_replication/tasks/todo.md` and corrections to `tasks/lessons.md`;
"proving it works" means running the relevant `VERIFICATION.md` stage (see above).

## Compute & training best practices

Apply these whenever writing or running training, eval, or data-generation code. The current
de-risking run is a single-GPU LoRA SFT, but **write code that scales to the multi-GPU / RL work**
without a rewrite.

**Parallelize as much as possible.**
- Make LLM and I/O calls **async and concurrent** — data generation (step 1), judge filtering
  (step 2), and eval should fire batched/concurrent requests, never a synchronous loop. Bound
  concurrency and back off on rate limits.
- **Data parallelism** (DDP / `accelerate` / `torchrun`) to shard the batch across GPUs, and
  **tensor parallelism** to split large models that don't fit on one card.
- Overlap data loading with compute (`num_workers`, prefetch, sequence packing) so the GPU never
  stalls waiting on the input pipeline.

**Use every memory/throughput lever.**
- **Flash attention** (`attn_implementation="flash_attention_2"`) for fast, memory-efficient
  attention — especially with the longer `framed_deliberation` targets.
- **Quantization** — 4-bit/8-bit QLoRA (`load_in_4bit`) when VRAM-bound; bf16 otherwise.
- **LoRA/PEFT** over full fine-tuning (already the default; keep it).
- Drive **GPUs to ~100% utilization** and **RAM/VRAM near full** — raise batch size / grad-accum /
  packing until the run is *compute*-bound, not idle. Profile (`nvidia-smi dmon`, torch profiler)
  to find and kill the bottleneck rather than guessing.

**Break gracefully — make everything resumable.**
- Checkpoint frequently and make every long job **resume from the last checkpoint**
  (`resume_from_checkpoint`); cache intermediate artifacts so a crash at hour 6 costs minutes, not
  the run. Assume RunPod spot pods get preempted.
- Prefer idempotent, re-runnable stages (like the committed data sidecars) over one-shot state.

**Observe everything.**
- Log **all** runs to **Weights & Biases** (`report_to="wandb"`) — loss, LR, grad norm, throughput,
  GPU/mem utilization, eval metrics, sample generations, and the full config. No un-tracked runs.
