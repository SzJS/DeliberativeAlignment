# Verification checklist

Run these on RunPod (the GPU stages need a GPU; data-prep stages are CPU-only and can run
anywhere). Each item = command + what to look for.

---

## Stage 1 — Data prep (CPU-only)

### 1.1 Inspect the workbooks
```bash
uv run python src/download_data.py
uv run python src/inspect_data.py
```
**Expect:** every `results/*.xlsx` lists sheets named `"{model} {approach}"` (e.g.
`DeepSeek-R1 RRC`). Final line:
`TOTAL DeepSeek-R1 RRC accuracy==1 across datasets: 319  (expected ~319)`.

### 1.2 Prepare data
```bash
uv run python src/prepare_data.py
```
**Expect (default: framed_deliberation, random split, DeepSeek-R1):**
- `training pool after malformed-drop: 367 (dropped 0)`  (the `rrc_only` condition instead
  gives `319` — the RRC accuracy==1 count from 1.1)
- `examples:  train=290 val=31 test(vignettes)=47`
- `test label base-rate (1=YES): ~0.47  (majority-class acc = ~0.53)`  ← stratified, ~balanced
- per-slice train counts populated for all four `(domain, difficulty)` cells
- **no assertion error** — the script hard-asserts no scenario-group overlap across splits
  (agent easy/hard share titles, so this guards real content leakage).
- prints `rendered length ... p95=...`; if p95 (tokens) > `max_seq_len`, raise `max_seq_len`
  (otherwise `train.py` will *drop* over-length examples rather than truncate the answer).

### 1.3 parse_output unit check (matches the paper's own labels)
```bash
uv run python - <<'PY'
import sys; sys.path.insert(0, "src")
import pandas as pd
from data_utils import parse_output
df = pd.read_excel("data/RRC_experiments/results/agent_hard_cases.xlsx", sheet_name="DeepSeek-R1 RRC")
ok = 0
for _, r in df.iterrows():
    b = parse_output(r["response_text"] if "response_text" in df else r["output"]).binary
    ok += int(b == r["output_binary"])
print(f"parse matches paper output_binary on {ok}/{len(df)} rows")
PY
```
**Expect:** matches on the large majority of rows (a few malformed generations may differ).

### 1.4 Eyeball rendered examples
```bash
uv run python - <<'PY'
import sys, json; sys.path.insert(0, "src")
r = json.loads(open("artifacts/train.jsonl").readline())
txt = " ".join(m["content"] for m in r["messages"])
print("system:", r["messages"][0]["content"][:80])
print("RRC spec leaked into input?", "virtual bargaining" in txt.lower())  # -> False
print("bare-answer instruction present?", "just YES or NO" in txt and "START_OUTPUT" not in txt)  # -> True
print("target answer:", r["answer"])
PY
```
**Expect:** spec **not** leaked (`False`), bare-answer instruction present (`True`).

---

## Stage 2 — Pipeline smoke test (cheap, 1.5B)

Set `smoke: true` in `config.yaml`, then:

### 2.1 Loss-mask check (do this before any training)
```bash
uv run python src/train.py --check
```
**Expect:** the `UNMASKED` block is the assistant target only —
`<think>…</think>\nYES` or `<think>…</think>\nNO` (+ eos) — and the `MASKED` block is the
system+user prompt. If the prompt tokens appear in `UNMASKED`, stop and fix before training.

### 2.2 Overfit-a-batch (train loop is correct)
```bash
uv run python src/train.py
```
**Expect:** with ~20 examples/1 epoch, train loss drops sharply toward ~0. Adapter saved to
`artifacts/adapter/`.

### 2.3 Eval runs end-to-end
```bash
uv run python src/evaluate.py
```
**Expect:** completes and writes `artifacts/metrics.json` + `eval_transcripts.jsonl`. Accuracy
is **not** meaningful at smoke scale — you're only checking it loads base+adapter, generates,
parses, and scores without crashing. Skim a transcript: generation should end in a bare
`YES`/`NO` after `</think>`.

Set `smoke: false` again for the real run.

---

## Stage 3 — Real run (7B)

```bash
bash scripts/run_all.sh
```
**Watch during training:** eval (val) loss — if it rises after ~epoch 2, the best-by-val
checkpoint is kept (`load_best_model_at_end`). Sample a generation and confirm well-formed tags.

**Read `metrics.json`:**
- **Success:** `sft.accuracy` ≈ `rrc_incontext.accuracy` (prompt-vs-distill on the same 7B) and
  `> no_thinking.accuracy`, at comparable/better `format_adherence`.
- **Per-slice:** check `agent/hard` and `development/hard` — thin hard-case training data shows
  up here. The default `framed_deliberation` already trains VB deliberation (+ an RRC preamble)
  on hard cases, so it's the richest condition; if hard slices lag, the lever is comparison, not
  a richer condition — rerun `condition: best_per_vignette` (same traces, no preamble) to check
  whether the framing preamble is helping or hurting, and note the number of hard `accuracy==1`
  VB traces is inherently limited.
- `paper_rrc.accuracy` is the paper's DeepSeek-R1 frontier number on the same held-out
  vignettes — a reference ceiling, not something the 7B distill is expected to match.
- Sanity: every method's `accuracy` should beat its `majority_baseline` to be meaningful.
