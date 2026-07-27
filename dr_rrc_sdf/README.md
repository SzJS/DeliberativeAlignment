# dr_rrc_sdf

Teach a **base** model about resource-rational contractualism through **synthetic documents**
(SDF, following Model Spec Midtraining), then fine-tune it on RRC reasoning traces (SFT,
following deliberative alignment) — and measure what the document stage bought.

> **Scaffold only.** Deps and docs are real; every pipeline module is a stub. Nothing has been
> run and the environment has never been resolved. See `tasks/todo.md`.

- Design, rationale, risks: **`experiment_description.md`**
- Broader project: `../project_description.md`
- Verification procedures (written, not yet executed): **`VERIFICATION.md`**

## The pipeline

| # | stage | what it does |
|---|---|---|
| 1 | `sdf_gen` | `assets/rrc_spec.md` → hierarchical document generation (domains → doc types → character assertions → ideas → documents) → `artifacts/sdf/corpus.jsonl` |
| 2 | `sdf_train` | continued pretraining on the corpus. **Plain next-token, no chat template, no loss mask** |
| 3 | `merge_sdf` | normalise the SDF output into `artifacts/models/merged_sdf/`. **Not optional** — SFT must initialise from merged weights, or a second LoRA silently trains against the base model |
| 4 | `sft_gen` | vignettes → spec-in-context CoTs → judge filter → blend public instruction data → leakage-safe splits |
| 5 | `sft_train` | LoRA SFT on the merged SDF checkpoint, completion-only loss, spec stripped |
| 6 | `merge_sft` | merge the SFT adapter → the arm's servable checkpoint |
| 7 | `eval` | `inspect_ai`: three tasks × three arms |

Stages 2 and 4 are deliberately asymmetric, and the asymmetry *is* the experiment. If either
acquires the other's mechanics, it is wrong — each has a `--check` that asserts the opposite of
the other's.

## Comparison arms

`sdf_sft` (full pipeline) · `sft_only` (no SDF — the ablation) · `sdf_only` (SFT's own
initialisation, so it is free, and it is what tells you whether SDF installed anything at all).
`base` and `spec_in_context` are implemented but off by default.

Both SFT arms train on the identical dataset with the same instruction mix at the same seed.

## Setup

```bash
uv sync                                 # CPU-only: both generation pipelines + eval authoring
uv sync --extra gpu                     # + training
uv sync --extra gpu --extra vllm        # + fast local serving for eval
```

The main GPU stack is anchored on **vLLM**, which pins `torch==2.11.0` exactly. That single
constraint sets the whole range; see the comment block in `pyproject.toml`.

### Optional second venv: unsloth

`unsloth` cannot share an environment with vLLM — it caps `transformers<=5.5.0` while vLLM
requires `>=5.5.3`, and it also caps `trl` below 1.0 and `datasets` below 5.0. So it gets its own
venv instead of its own compromise:

```bash
UV_PROJECT_ENVIRONMENT=.venv-unsloth uv sync --extra gpu-unsloth
bash scripts/run_all.sh --train-venv .venv-unsloth --set model.backend=unsloth
```

Training runs there (unsloth's LoRA kernels: ~2× faster, materially lower VRAM); vLLM stays in
the default venv and inspect reaches it over HTTP (`eval.serve: vllm_server`), so the two never
need to meet. Both venvs pin `torch 2.11`, so checkpoints written by one load in the other.

**Scope:** unsloth is LoRA-only. The SDF stage defaults to full-parameter FT, which it does not
accelerate — this buys you the SFT stage and the LoRA SDF fallback. If you never run LoRA, skip
the second venv entirely.

**None of these ranges has been resolved yet.** The first `uv sync --extra gpu` on a GPU host is
the real test; commit the resulting `uv.lock` (which, unlike in `dr_rrc_replication`, is *not*
git-ignored here).

## Run

```bash
bash scripts/preflight.sh                       # checks secrets, spec, GPU, disk. No API calls.
bash scripts/run_all.sh                         # the sdf_sft arm
bash scripts/run_all.sh --arm sft_only          # the ABLATION arm (SDF stages auto-skipped)
bash scripts/run_all.sh --from sdf_train        # resume after a crash
bash scripts/run_all.sh --only eval
bash scripts/run_all.sh --set sdf_train.epochs=1
bash scripts/run_all.sh --dry-run               # print the plan, run nothing
```

**Training both arms is two invocations.** `--arm` sets `sft_train.init_from` *and* scopes the
SFT/merge/eval stamps, so the second invocation neither collides with nor overwrites the first.

Every stage is independently resumable at two levels: coarse stamps in `artifacts/.stamps/` skip
completed stages, and each stage's own mechanism (response cache / `resume_from_checkpoint` /
inspect `eval_set` log dir) means a crash costs minutes rather than the run. A stage stamped
under a *different* config refuses to be skipped rather than warning. Unknown stage names in
`--only`/`--skip`/`--from` are rejected rather than silently running everything or nothing.

Generation stages take `--dry-run` to render and price prompts without making a single API call.

## How to read the results

**`rrc_decision` accuracy is not the headline metric.** The paper's vignette set is separable by
difficulty — all hard cases are ground-truth YES, all easy NO — so difficulty alone predicts the
label, and YES/NO accuracy cannot distinguish real deliberation from a difficulty heuristic.

Read, in order:

1. **`spec_recall` on `sdf_only`** — did the documents install anything? If not, everything
   downstream is uninformative rather than negative. This is a **gate**: run it before paying for
   the SFT stage.
2. **`rrc_procedure`, `sdf_sft` vs `sft_only`** — the actual question. Report the stderr on the
   *difference*, not just on each arm.
3. **`rrc_decision`** — with `format_adherence` beside it, and each arm against its own
   majority-class baseline. An arm that does not beat its baseline is not a result.

## Outputs

`artifacts/` — `sdf/`, `sft/`, `models/`, `eval/logs/`, `eval/report.md`, `.stamps/`.
`outputs/` — the response cache, every generation as jsonl, and every stage's tee'd terminal log.
Both git-ignored.

## Data provenance

- Vignettes (`vignette_source: paper`): <https://github.com/mint-philosophy/RRC_experiments> (MIT).
- Chat template: vendored from `ibm-granite/granite-4.1-8b` (Apache 2.0) — see `assets/README.md`.
- Base model: `ibm-granite/granite-4.1-8b-base` (Apache 2.0); `granite-4.1-3b-base` for smoke.
- Public instruction mix: **not yet chosen** — record the dataset and its licence here.
