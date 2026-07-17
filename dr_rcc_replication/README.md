# dr_rcc_replication

Replicate the RCC/RRC moral-reasoning experiment with a **trained** model instead of a
**prompted** one. We SFT a small DeepSeek-R1-Distill model on the paper's own RRC reasoning
traces, with the RRC spec **removed** from the prompt (deliberative-alignment *context
distillation*), so the model recalls and runs the procedure from a bare prompt.

- Source data: <https://github.com/mint-philosophy/RRC_experiments> (MIT).
- Broader project + design rationale: `../project_description.md`, `experiment_description.md`,
  and the plan at `~/.claude/plans/hey-claude-i-want-calm-wilkes.md`.

## What the pipeline does

1. **download_data** — clone the RRC_experiments repo into `data/`.
2. **inspect_data** — confirm the workbook layout (sheets named `"{model} {approach}"`).
3. **prepare_data** — pull DeepSeek-R1 **RRC**, `accuracy == 1` traces (~319), split on
   **scenario key** (leakage-safe), write `train/val/test.jsonl` + `splits.json`.
4. **train** — LoRA SFT (unsloth if available, else transformers), bf16, completion-only loss.
   Reasoning target uses the base model's native `<think>…</think>`; the answer uses
   `START_OUTPUT YES/NO END_OUTPUT`.
5. **evaluate** — score the SFT model vs baselines on the frozen test vignettes.

## Key design points

- **DeepSeek-R1 only** by default: the base (`DeepSeek-R1-Distill-Qwen`) was distilled from
  DeepSeek-R1, so those traces go with the grain. Expand via `condition: best_per_vignette`
  (adds virtual-bargaining traces on hard cases) if the ~319 prove thin — see `config.yaml`.
- **Context distillation**: SFT input carries a minimal system prompt only; `strip_spec: true`.
- **Fair test set**: built from the *unfiltered* held-out scenarios, so the `paper_rrc`
  baseline (the paper's own answer on those stories) is a real number, not a trivial 1.0.
- **Split caveat**: `split_mode: stakes_generalisation` yields an all-YES hard test set
  (majority baseline 1.0). Read its accuracy together with the easy `val` (all-NO) to catch a
  YES-collapse — accuracy alone can't distinguish real deliberation from a YES-bias there.

## Setup (RunPod)

```bash
# CPU-only steps (data prep) work with just the base deps:
uv sync
# GPU training/eval:
uv sync --extra gpu                 # add --extra unsloth for faster LoRA
# If the RunPod image ships its own CUDA torch, expose it to uv:
#   uv venv --system-site-packages && uv sync --extra gpu
```

## Run

```bash
bash scripts/run_all.sh                 # full run with config.yaml
# or step-by-step:
uv run python src/download_data.py
uv run python src/prepare_data.py
uv run python src/train.py --check      # print the loss mask, then:
uv run python src/train.py
uv run python src/evaluate.py           # -> artifacts/metrics.json
```

**Smoke test first** (cheap 1.5B pipeline check): set `smoke: true` in `config.yaml`, then
`bash scripts/run_all.sh`. See `VERIFICATION.md` for the full checklist and expected outputs.

## Outputs (`artifacts/`)

`train/val/test.jsonl`, `splits.json`, `adapter/` (LoRA), `metrics.json`,
`eval_transcripts.jsonl`.

## Notes

- Versions in `pyproject.toml` are ranges; pin to the RunPod base image if it differs.
- `unsloth` is version-sensitive (torch/triton); `train.py` auto-falls back to the plain
  transformers path if it can't import, so the run never hard-blocks on it.
