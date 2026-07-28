# dr_rrc_sdf — Design Handoff

> **Status: scaffold only.** The dependency and documentation layer is real; every pipeline
> module is a stub. Nothing here has been executed — no training, no data download, no API call.
> The environment has never been resolved. See `tasks/todo.md` for what remains.

## 1. Why this experiment

`dr_rrc_replication` answered a narrow question: can a model be *trained* to run the RRC
procedure rather than *prompted* with it? It did so by SFT-ing an instruct model
(DeepSeek-R1-Distill) on the RRC paper's own published reasoning traces, with the spec stripped
from the prompt.

That leaves the more interesting question untouched. Deliberative alignment's step 3 is context
distillation — the model must recall the procedure from weights. But *where do the weights learn
what RRC is?* In `dr_rrc_replication`, only from a few hundred demonstrations. Demonstrations
underspecify the *reason* a behaviour is correct, so the model can learn the surface pattern
without the intended generalisation — which is precisely the critique Model Spec Midtraining
makes of standard alignment fine-tuning.

MSM's answer is to insert a phase *before* alignment fine-tuning in which the model trains on
synthetic documents that discuss the spec, as ordinary next-token prediction. The spec supplies
the intended generalisation in natural language up front; fine-tuning then elicits and reinforces
it. **This experiment is that phase, applied to RRC.**

The question: **does teaching a model about resource-rational contractualism through documents,
before showing it any demonstrations, change what it learns from those demonstrations?**

## 2. The pipeline

```
             assets/rrc_spec.md
                    |
      (1) hierarchical document generation          [OpenRouter, async]
                    |
             ~2k synthetic documents
                    |
      (2) continued pretraining                     [next-token, NO mask, NO chat template]
                    |
             merged_sdf  ────────────────────────────────┐
                    |                                    |
      (3) CoT generation + judge filter             [OpenRouter, async]
             + public instruction mix
                    |
      (4) SFT                                       [LoRA, completion-only loss]
                    |
             merged_sdf_sft                              |
                    |                                    |
      (5) inspect_ai eval  ←── merged_sft_only ←─────────┘ (same SFT, no SDF)
```

Stage 3 follows the RRC paper's own method for producing SFT data — spec in context, model
reasons over it, judge filters — rather than reusing the paper's published traces. Stage 4
follows deliberative alignment's step 3 (spec stripped). **RL, deliberative alignment's step 4,
is deliberately out of scope**, as in experiment 1, to cut cost.

## 3. What changed from experiment 1, and why

| | `dr_rrc_replication` | `dr_rrc_sdf` |
|---|---|---|
| Base | DeepSeek-R1-Distill-Qwen-7B (instruct, reasoning-distilled) | `granite-4.1-8b` (**instruct**) |
| Stages | SFT only | **SDF → SFT** |
| SFT data | the paper's published traces | **generated + judge-filtered by us**, plus MSM's Table 2 instruction mix |
| Eval | hand-rolled `evaluate.py` | **`inspect_ai`** |
| Accelerator | unsloth (pinned the whole stack) | vLLM anchors; unsloth optional, in its **own venv** |

**Why an instruct model.** This reverses an earlier decision to build on `granite-4.1-8b-base`,
and the reversal follows MSM. Their headline safety result (§4–5) midtrains *production Instruct*
models — Qwen2.5-32B-Instruct and Qwen3-32B — and reports the agentic-misalignment reduction
(54%→7% on Qwen3-32B) on those. Only the small cheese-preference study in their §3 uses a base
model. The result we are trying to transfer is the §4–5 one, so we should match its setting.

The cost is real and worth stating plainly, because it is exactly what the earlier base-model
argument was guarding against: SDF is a pretraining-style intervention, and running it on an
already-aligned checkpoint means the documents compete with whatever alignment training that
checkpoint received. "What did the documents install?" is a harder question when the model's
priors about how to behave are not blank.

Three things make that acceptable:

- The **`sdf_only` arm measures exactly this.** It evaluates SDF's own checkpoint, so an
  above-floor `spec_recall` is direct evidence the documents installed something the instruct
  model did not already have. That arm was already in the design and is free; the model switch
  promotes it from a sanity check to the load-bearing control.
- **Both SFT arms train on an identical dataset at an identical seed**, so whatever the instruct
  checkpoint brings, it is held constant across the `sdf_sft` vs `sft_only` contrast that SDF is
  actually measured by.
- **MSM demonstrated this setting works** on models that had already been alignment-trained, which
  is the empirical answer to the objection.

The benefit is that we no longer have to install general instruction-following from scratch on top
of a few hundred RRC vignettes — a demand the base model made, and one the RRC dataset is far too
small to meet on its own.

**Why instruction data in SFT, not SDF.** The conclusion is unchanged by the switch; the *reason*
is not. It is no longer "the base model cannot follow instructions at all". It is now MSM's own
§4–5 rationale: midtraining on a narrow synthetic corpus degrades an Instruct model's general
coherence, and a fixed instruction mix in the subsequent fine-tune repairs it. MSM used 2M tokens
for precisely this, and said so — "mostly to fix incoherence caused by midtraining on Instruct
models" (Appendix B.3). We take their Table 2 mix verbatim; see `config.yaml`
`sft_data.instruction_mix` and the provenance table in `README.md`.

The mechanism argument stands on its own either way: the SDF corpus is trained exactly like
pretraining data — that is MSM's mechanism claim — and injecting chat formatting there would
defeat it. A separate knob (`sdf.replay`) mixes generic *raw text* into SDF to counter forgetting;
that is a different thing.

**What the switch does not change.** `granite-4.1-8b` and `granite-4.1-8b-base` are the same
architecture: 40 layers, hidden 4096, vocab 100352, `tie_word_embeddings: true`,
`logits_scaling: 16.0`, `pad` (100256) ≠ `eos` (100257). Every Granite invariant in CLAUDE.md
carries over untouched. The one practical difference is that the instruct checkpoint ships its own
chat template, so the vendored `assets/granite_chat_template.jinja` stops being *required* and
becomes purely a reproducibility anchor — `data_utils.install_chat_template` overwrites the
model's own with the vendored one and warns when they differ, which is the behaviour we want:
an upstream template edit must not silently re-render every training example.

## 4. Comparison arms

| arm | what it is | what it answers |
|---|---|---|
| `sdf_sft` | full pipeline | the treatment |
| `sft_only` | SFT from the untouched `granite-4.1-8b` instruct checkpoint, no SDF | **what did SDF buy?** |
| `sdf_only` | SFT's own init checkpoint | **did SDF install anything at all?** (free — the checkpoint already exists) |
| `base` | untrained (available, off) | floor |
| `spec_in_context` | spec prompted, untrained (available, off) | prompt vs train |

Both SFT arms train on the **identical** dataset, same instruction mix, same seed. Otherwise the
comparison measures instruction data rather than SDF.

## 5. What "success" looks like

`sdf_sft` beats `sft_only` on **procedure fidelity** — selecting the right approximation and
executing it — at equal or better decision accuracy, with `sdf_only` showing above-floor
`spec_recall` to confirm the documents installed something. Report the stderr on the *difference*
between arms, not just on each arm.

A clean negative result is publishable and should be pre-registered as such: if `sdf_sft` ≈
`sft_only` **and** `sdf_only` shows real spec recall, that says document-based spec training does
not transfer to procedural behaviour at this scale — a finding. If `sdf_only` shows *no* spec
recall, that is not a finding about RRC at all (see risk 1).

## 6. Known risks

**1. SDF may install nothing at de-risking scale — and this is the risk that makes the whole
experiment uninformative rather than negative.** MSM operates at pretraining scale. A continued
pretrain over a few million synthetic tokens may produce no measurable effect, in which case
`sdf_sft ≈ sft_only` is an *uninformative null* (the intervention didn't take) rather than a
result about RRC.

*Mitigations, decided up front:* full-parameter FT rather than LoRA for the SDF stage (the
config default; LoRA may lack the capacity to install knowledge as opposed to style — hence
r=64 there vs r=16 for SFT); multiple epochs; and **decisively, `spec_recall` on the `sdf_only`
checkpoint is a GATE before the SFT stage is paid for.** Find out while it is cheap to change.

**2. Catastrophic forgetting** from continued pretraining on a narrow synthetic corpus.
Mitigation: the `sdf.replay` generic-text mix, a config knob from day one (suggest 20–50%) rather
than a retrofit.

**3. Corpus diversity collapse.** The hierarchy exists to prevent it, but at de-risking scale
with few domains the documents will still tend toward paraphrase. `filter_corpus` needs
near-duplicate detection, and `stats.json` must report a diversity number that gets **looked at
before training** — this failure is invisible in a loss curve.

**4. Generation cost is unbudgeted.** ~2k documents × ~800 words is several million output
tokens, plus stages 1–4 and the CoT samples. `--dry-run` exists to price a run without spending;
`generation.max_cost_usd` is a hard stop.

**5. The spec-identity confound.** If `assets/rrc_spec.md` (the SDF seed) and the in-context spec
are different documents, the model was taught spec A and evaluated against spec B. Resolution:
`rrc_spec.md` is a superset containing `prompts.RRC_PROMPT` verbatim, asserted at load. Not yet
implemented — `rrc_spec.md` is a placeholder.

**6. Inherited from experiment 1 and still true.** The paper's vignettes are separable by
difficulty: all hard cases are ground-truth YES, all easy NO. So difficulty alone predicts the
label, and decision accuracy can never show that deliberation *content* helps. This is why
`rrc_procedure` and `spec_recall` exist, and why `rrc_decision` **must not be the headline
metric**. Under `split_mode: stakes_generalisation` the hard test set is all-YES, majority
baseline 1.0 — read it against the easy `val` (all-NO) to catch a YES-collapse.

**7. Format fairness.** The untrained arms have never seen our `<reasoning>`/`ANSWER:` format, so
a low score from them may be a formatting failure rather than a reasoning one. Mitigated with
format-only few-shot exemplars and by always reporting `format_adherence` beside accuracy.

## 7. Open items

- [ ] Write `assets/rrc_spec.md` (blocks everything downstream). Decide superset vs verbatim.
- [ ] Pick the OpenRouter models per generation stage, and price a run with `--dry-run`.
- [ ] Pick the public instruction dataset (licence + size) and the RRC:instruction ratio.
- [ ] Pick the `sdf.replay` dataset and fraction.
- [ ] Resolve the ground-truth-label question for `vignette_source: generated` (see
      `generation/vignettes_generated.py`) — currently why `paper` is the default.
- [ ] Decide filter-vs-rewrite for spec-leakage artifacts in judged CoTs.
- [ ] Resolve the environment: `uv sync --extra gpu --extra vllm` on RunPod, then commit
      `uv.lock`. The pin ranges are reasoned but **unverified**.
- [ ] Sweep the learning rate. Granite's `logits_scaling: 16.0` means experiment 1's `1e-4` does
      not transfer.
- [ ] Update the root `CLAUDE.md`: its Layout section still describes `dr_synthetic/` (deleted in
      `aa1bc9a`) and its invariants are experiment-1-scoped (`<think>`, `START_OUTPUT`, unsloth).
