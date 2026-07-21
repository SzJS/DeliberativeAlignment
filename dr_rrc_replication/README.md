# dr_rrc_replication

Replicate the RRC (Resource-Rational Contractualism) moral-reasoning experiment with a
**trained** model instead of a **prompted** one. We SFT a small DeepSeek-R1-Distill model on
the paper's own RRC reasoning traces, with the RRC spec **removed** from the prompt
(deliberative-alignment *context distillation*), so the model recalls and runs the procedure
from a bare prompt.

The directory and Python import root are `dr_rrc_replication`; only the `pyproject.toml`
`[project].name` still reads `dr-rcc-replication` (cosmetic — `package = false`).

- Source data: <https://github.com/mint-philosophy/RRC_experiments> (MIT).
- Broader project + design rationale: `../project_description.md`, `experiment_description.md`.
- Full architecture, invariants, and per-module notes: `../CLAUDE.md`.

## What the pipeline does

1. **download_data** — clone the RRC_experiments repo into `data/`.
2. **inspect_data** — confirm the workbook layout (sheets named `"{model} {approach}"`).
3. **prepare_data** — pull DeepSeek-R1 traces for the configured `condition`, keep the
   `accuracy == 1` ones, split on **scenario key** (leakage-safe), and write
   `train/val/test.jsonl` + `splits.json`. The test set is built from the **unfiltered**
   held-out vignettes, so accuracy and the `paper_rrc` baseline are real numbers.
4. **train** — LoRA SFT (unsloth if available, else transformers), bf16, completion-only loss.
   Reasoning target uses the base model's native `<think>…</think>`; the answer is a bare
   `YES`/`NO`.
5. **evaluate** — score the SFT model vs baselines (`no_thinking`, `rrc_incontext`,
   `paper_rrc`) on the frozen test vignettes.

## Training conditions (`condition:` in `config.yaml`)

Which traces become SFT targets. Default is **`framed_deliberation`**.

- **`rrc_only`** — the paper's RRC-approach traces (rule-based CoT), `accuracy == 1`.
- **`best_per_vignette`** — one accurate trace per vignette, preferring virtual-bargaining
  (VB) deliberation on hard cases and rules on easy cases. Single-model, for CoT consistency.
- **`framed_deliberation`** *(default)* — `best_per_vignette`, but each selected hard VB
  deliberation gets a short first-person RRC procedure-selection preamble prepended ("this
  case is high-stakes, so I'll simulate what the stakeholders would agree to…"), then the full
  VB deliberation **verbatim**. Only the preamble is LLM-written; the VB body and the YES/NO
  answer are deterministic code. The preambles are committed to
  `assets/framed_deliberation.json` — the reproducibility anchor, since `data/` is git-ignored;
  rebuild with `scripts/build_framed_sidecar.py`. Requires `split_mode: random` (hard traces
  are test-only under `stakes_generalisation`, so framing would be a silent no-op).

With the default config (`framed_deliberation`, `random` split, `seed: 0`) `prepare_data`
currently writes **~290 train / 31 val / 47 test** examples. Treat this as a quickstart
sanity check — the exact counts move with `condition`, `split_mode`, and `seed`. Detailed
per-slice expectations live in `VERIFICATION.md`.

## Key design points

- **DeepSeek-R1 traces by default**: the base (`DeepSeek-R1-Distill-Qwen-7B`) was distilled
  from DeepSeek-R1, so those traces go with the grain. (`models` picks whose result sheets to
  pull; `condition` picks which traces — two independent axes.)
- **Context distillation**: SFT input carries a minimal system prompt only (`strip_spec: true`).
  The RRC spec must not leak into the input — the model recalls the procedure from weights.
- **Fair test set**: built from the *unfiltered* held-out scenarios, so the `paper_rrc`
  baseline (the paper's own answer on those stories) is a real number, not a trivial 1.0. The
  default `random` split is stratified, giving a roughly balanced test set (base rate ~0.47).
- **Split caveat**: if you switch to `split_mode: stakes_generalisation` (train easy/low-stakes,
  test hard/high-stakes), the hard test set is all-YES (majority baseline 1.0). Read its
  accuracy together with the easy `val` (all-NO) to catch a YES-collapse — accuracy alone can't
  distinguish real deliberation from a YES-bias there.

## Setup (RunPod)

```bash
# CPU-only steps (data prep) work with just the base deps:
uv sync
# GPU training/eval:
uv sync --extra gpu                 # add --extra unsloth for faster LoRA
# If the RunPod image ships its own CUDA torch, expose it to uv:
#   uv venv --system-site-packages && uv sync --extra gpu
```

`attn_implementation: flash_attention_2` (config default) applies only to the transformers
fallback path — unsloth uses its own kernels, and the fallback auto-degrades to `sdpa` if
flash-attn/Ampere is absent. flash-attn isn't in the extras; install it manually if you want it:
`uv pip install flash-attn --no-build-isolation`.

## Run

```bash
bash scripts/run_all.sh                             # full run with config.yaml
# or step-by-step (each takes --config):
uv run python src/download_data.py --config config.yaml
uv run python src/inspect_data.py  --config config.yaml
uv run python src/prepare_data.py  --config config.yaml
uv run python src/train.py --check --config config.yaml   # print the loss mask, then:
uv run python src/train.py         --config config.yaml
uv run python src/evaluate.py      --config config.yaml    # -> artifacts/metrics.json
```

`run_all.sh` accepts an alternate config: `bash scripts/run_all.sh other.yaml`.

**Smoke test first** (cheap 1.5B pipeline check): set `smoke: true` in `config.yaml`, then
`bash scripts/run_all.sh`. See `VERIFICATION.md` for the full checklist and expected outputs.

## Outputs (`artifacts/`)

`train/val/test.jsonl`, `splits.json`, `adapter/` (LoRA), `metrics.json`,
`eval_transcripts.jsonl`.

## Notes

- Versions in `pyproject.toml` are ranges; pin to the RunPod base image if it differs.
- `unsloth` is version-sensitive (torch/triton); `train.py` auto-falls back to the plain
  transformers path if it can't import, so the run never hard-blocks on it.
