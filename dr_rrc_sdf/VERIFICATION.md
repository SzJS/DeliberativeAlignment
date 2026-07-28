# Verification checklist

> **NOTHING IN THIS FILE HAS BEEN EXECUTED.** It was written alongside the scaffold, on a
> local machine with limited RAM and no GPU, so that a later GPU session has a checklist rather
> than a blank page. Every "Expect" below is a *prediction* to be confirmed, not an observation.
> Correct them as you go — a wrong expectation recorded here is worse than none.

Each item = command + what to look for. Stage 0 runs anywhere; stages 1–5 follow the pipeline.

---

## Stage 0 — Scaffold sanity (CPU, no network, free)

### 0.1 Everything parses and imports
```bash
uv sync
uv run python -c "import dr_rrc_sdf, dr_rrc_sdf.data_utils, dr_rrc_sdf.prompts; print('ok')"
uv run python -c "import yaml; yaml.safe_load(open('config.yaml')); print('config ok')"
bash -n scripts/run_all.sh && bash -n scripts/lib.sh && bash -n scripts/preflight.sh && echo 'shell ok'
```
**Expect:** all four succeed. The import test is the one that matters — it proves the package is
installed editable and that **no `PYTHONPATH` shim is needed**. If it fails, run `uv sync`; do
not add a path hack (see the comment in `run_all.sh`).

### 0.2 The RRC spec is byte-identical to experiment 1's
```bash
uv run python - <<'PY'
import ast, hashlib
def grab(p, n):
    for node in ast.parse(open(p).read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t,'id',None)==n for t in node.targets):
            return ast.literal_eval(node.value)
a = grab('../dr_rrc_replication/src/prompts.py', 'RRC_PROMPT')
b = grab('src/dr_rrc_sdf/prompts.py', 'RRC_PROMPT')
print('identical:', a == b, '| sha:', hashlib.sha256(b.encode()).hexdigest()[:12])
PY
```
**Expect:** `identical: True`, `sha: b50a1b7b95ec`. *(Confirmed at scaffold time — this is the one
check in this file that HAS been run.)* One spec, three consumers; see the docstring in
`prompts.py`. If this ever goes False, find out which consumer moved before running anything.

### 0.3 Config overrides reach nested sections and reject typos
```bash
uv run python - <<'PY'
from dr_rrc_sdf.config_utils import load_config, resolve, set_by_path
cfg = load_config('config.yaml', snapshot=False)
print('epochs:', resolve(cfg, 'sdf_train.epochs'))
try:
    set_by_path(cfg, 'sdf_train.epocs', 1); print('BUG: typo accepted')
except KeyError as e:
    print('typo rejected ok:', e)
PY
```
**Expect:** the epochs value prints, and the typo raises. This is the bug class
`dr_rrc_replication`'s flat `cfg.update()` allows silently — a mistyped smoke override there does
nothing, and the "smoke" run trains for the real number of epochs.

### 0.4 Preflight fails loudly on the placeholder spec
```bash
bash scripts/preflight.sh
```
**Expect (today):** `[FAIL] assets/rrc_spec.md is still the placeholder` and
`[FAIL] OPENROUTER_API_KEY is EMPTY` — the repo-root `.env` is currently a 0-byte file. Both are
correct failures at this point; preflight exists so they surface before a run spends anything.

---

## Stage 1 — SDF corpus generation

### 1.1 Price the run before paying for it
```bash
uv run python -m dr_rrc_sdf.generation.sdf_pipeline --dry-run
```
**Expect:** rendered sample prompts, an estimated token count and cost, and **zero API calls**.
Compare the estimate against `generation.max_cost_usd` before proceeding. ~2k documents × ~800
words is several million output tokens — this is the largest single cost in the pipeline.

### 1.2 Review the manifest before generating documents
```bash
uv run python -m dr_rrc_sdf.generation.sdf_pipeline --emit-manifest
```
**Expect:** `assets/sdf_manifest.json` populated with domains/subdomains/doc types/assertions/
ideas, and the pipeline stops. Stage 5 is ~99% of the cost, so read the manifest first: are the
domains broad and non-overlapping (MSM biases toward *fewer, broader*)? Do the document types
look like things that would really exist on the internet? Are the ideas genuinely different
perspectives, or the same document five times?

### 1.3 Generate, then check diversity BEFORE training
```bash
uv run python -m dr_rrc_sdf.generation.sdf_pipeline
cat artifacts/sdf/stats.json
```
**Expect:** documents kept/dropped per filter reason, and a **diversity number**. Diversity
collapse is the main failure mode of hierarchical generation at de-risking scale and it is
**invisible in a loss curve** — this is the only place it shows up. Also skim ~10 documents by
hand for the constraints that filters catch poorly: fabricated dates/authors/citations/URLs, and
values or motivations not present in the spec.

---

## Stage 2 — SDF training

### 2.1 Loss-mask check — the INVERSE of the SFT one
```bash
uv run python -m dr_rrc_sdf.training.train_sdf --check
```
**Expect:** a decoded packed block of **raw document prose**, containing **no**
`<|start_of_role|>`, and the assertion `labels == input_ids` (except padding) passing.

This is continued pretraining. If any token is masked, or the chat template has leaked in, stop.
Run this together with 4.1 — two stages, two opposite assertions, and running both is how you
prove neither has drifted into the other's mechanics.

### 2.2 Train
```bash
uv run python -m dr_rrc_sdf.training.train_sdf 2>&1 | tee outputs/logs/sdf_train.manual.log
# multi-GPU / full-parameter:
# accelerate launch -m dr_rrc_sdf.training.train_sdf
# LoRA via unsloth (separate venv — it cannot coexist with vllm):
# UV_PROJECT_ENVIRONMENT=.venv-unsloth uv run python -m dr_rrc_sdf.training.train_sdf \
#     --set model.backend=unsloth --set sdf_train.full_finetune=false
```
**Expect if using the unsloth backend with `full_finetune: true`:** a loud message that unsloth is
being ignored for this stage (it is LoRA-only). Silence there would mean you believe a full-FT run
was accelerated when it wasn't.
**Expect:** loss decreasing smoothly. Watch for degeneration — sample generations should stay
coherent English. If they don't, raise `sdf.replay.fraction` (catastrophic forgetting on a narrow
corpus is the predicted cause). Confirm GPU utilisation is near 100%: if not, packing or the
input pipeline is the bottleneck, not the model.

### 2.3 Merge — a pipeline stage, not an afterthought
```bash
uv run python -m dr_rrc_sdf.training.merge --stage sdf     # run_all.sh does this as `merge_sdf`
```
**Expect:** `artifacts/models/merged_sdf/` with weights **and** tokenizer **and** chat template.
Skipping this makes the SFT stage silently train against the base model, producing a clean run
and a null SDF result that is actually a plumbing bug. There is no error message for that failure.

### 2.4 ⚠️ GATE — did SDF install anything?
```bash
uv run python evals/run_evals.py --arms sdf_only --tasks spec_recall
```
**Expect:** `sdf_only` measurably above the `base` floor on spec recall.

**If it is not, stop and fix the SDF stage before paying for SFT.** An `sdf_sft ≈ sft_only`
result at the end of a run where SDF installed nothing is *uninformative*, not negative — and
this is the cheap moment to discover it. Levers: full FT instead of LoRA, more epochs, larger
corpus, higher LoRA rank.

Sanity check in the other direction: if the **`base`** arm also scores well, the probes are
answerable from general ethical knowledge and are not measuring the spec at all.

---

## Stage 3 — SFT data generation

```bash
uv run python -m dr_rrc_sdf.generation.sft_pipeline --dry-run   # price first
uv run python -m dr_rrc_sdf.generation.sft_pipeline
```
**Expect:**
- judge **accept rate** reported. A very high accept rate means the rubric is not discriminating,
  not that generation was excellent.
- **spec-leakage** hits reported separately. CoTs generated with the spec in context but trained
  on with it stripped say "as the policy above states" — a dangling reference to a document that
  won't exist at inference. If most traces trip this, fix the generation prompt rather than
  filtering 80% of the data away.
- `train / val / test` counts, with the **instruction-mix count reported separately** from the RRC
  count.
- test label base rate and majority-class accuracy.
- p95 rendered token length ≤ `model.max_seq_len`. If it exceeds, **raise `max_seq_len`** —
  over-length examples are dropped, never truncated, because truncation would cut the answer and
  eos and train an answer-less, non-terminating target.
- **no assertion error** from the scenario-group overlap check in `make_splits`.

### 3.1 Spot-check leakage safety and spec stripping
```bash
uv run python - <<'PY'
import json
from dr_rrc_sdf.prompts import RRC_PROMPT
r = json.loads(open('artifacts/sft/train.jsonl').readline())
txt = ' '.join(m['content'] for m in r['messages'])
print('spec leaked into input?', 'virtual bargaining' in txt.lower())   # -> False
print('answer instruction present?', 'ANSWER: YES' in txt)              # -> True
splits = json.load(open('artifacts/sft/splits.json'))
print('scenario overlap:', set(splits['train']) & set(splits['test']))  # -> set()
PY
```
**Expect:** `False`, `True`, `set()`. With `strip_spec: true` the RRC spec must not reach the SFT
input — the model is meant to recall the procedure from weights, and a leak silently converts the
experiment into a prompted baseline.

---

## Stage 4 — SFT training

### 4.1 Loss-mask check — the INVERSE of 2.1
```bash
uv run python -m dr_rrc_sdf.training.train_sft --check
```
**Expect:** `UNMASKED` is the assistant target **only** —
`<reasoning>…</reasoning>\n\nANSWER: YES` plus eos — and `MASKED` is the system+user prompt. If
prompt tokens appear in `UNMASKED`, stop and fix before training.

### 4.2 Train both arms
Two full invocations — `--arm` sets `sft_train.init_from` and scopes the stamps, so the second
does not collide with the first:
```bash
bash scripts/run_all.sh                    # sdf_sft:  base -> SDF -> merge -> SFT -> merge
bash scripts/run_all.sh --arm sft_only     # sft_only: base -> SFT -> merge (SDF auto-skipped)
```
**Expect:** both arms train on the **identical** dataset (same instruction mix, same seed) —
confirm the reported example counts match exactly. If they differ, the comparison measures data
rather than SDF.

Watch the learning rate: `config.yaml` starts at `1e-5`, **not** experiment 1's `1e-4`. Granite's
`logits_scaling: 16.0` flattens the softmax, so loss magnitude and effective LR do not behave like
a Llama-shaped model's. If loss is flat or diverging, sweep before concluding anything.

---

## Stage 5 — Eval

```bash
uv run python evals/run_evals.py --config config.yaml
inspect view --log-dir artifacts/eval/logs      # per-sample transcripts
cat artifacts/eval/report.md
```

**Read the report in this order** (see README, "How to read the results"):

1. `spec_recall` on `sdf_only` — already checked at gate 2.4; confirm it held.
2. `rrc_procedure`, `sdf_sft` vs `sft_only` — **the actual question.** Report the stderr on the
   *difference* between arms, not just on each arm; two overlapping error bars get read as "no
   effect" when the paired difference is often significant, and vice versa.
3. `rrc_decision` — **not** the headline. The vignette set is separable by difficulty (all hard
   are YES, all easy NO), so accuracy alone cannot show that deliberation *content* helps. Read it
   with `format_adherence` beside it and each arm against its own majority-class baseline.

**Sanity checks:**
- Every arm's merged checkpoint must exist before launch — a missing one must fail fast, not get
  silently evaluated as some other model under the wrong arm label.
- Under `split_mode: stakes_generalisation` the hard test set is all-YES (majority baseline 1.0).
  Read it together with the easy `val` (all-NO) to catch a YES-collapse.
- If the untrained arms are enabled, their low scores may be *formatting* failures — check
  `format_adherence` before reading them as reasoning failures.

---

## Smoke run (write this down, run it first on a GPU host)

Set `smoke.enabled: true`, which swaps to `granite-4.1-3b` — same architecture, same
tokenizer, **same role tokens**, so the format code path is identical to the real run (unlike
experiment 1's 1.5B↔7B swap) — and shrinks every stage. Note it shrinks **all four** SDF
hierarchy knobs (2×2×2×2 = 16 documents vs 8×4×5×6 = 960): corpus size is their product, so
shrinking one would leave a "smoke" run costing nearly a real one.

```bash
bash scripts/run_all.sh --force
```
**Expect:** the whole pipeline executes end to end. **Accuracy is meaningless at this scale** —
you are only checking that it generates, parses, packs, masks correctly, trains, merges, serves,
and scores without crashing. Then set `smoke.enabled: false` and clear `artifacts/` before the
real run (`run_all.sh` will refuse to skip stages stamped under a different config, which is the
mechanical version of that instruction).
