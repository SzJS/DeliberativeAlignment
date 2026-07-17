# dr_rcc_replication — build todo

Plan: `/home/jazon/.claude/plans/hey-claude-i-want-calm-wilkes.md`

## Build
- [x] Project scaffolding: `pyproject.toml` (uv), `config.yaml`, `.gitignore`
- [x] `src/prompts.py` — verbatim paper prompts + tag constants
- [x] `src/data_utils.py` — `parse_output`, message/target builders
- [x] `src/datasets_common.py` — xlsx schema, sheet naming, normalized loader
- [x] `src/download_data.py` — clone RRC_experiments into `data/`
- [x] `src/inspect_data.py` — enumerate sheets/columns (sanity)
- [x] `src/prepare_data.py` — load DeepSeek RRC sheets → filter → scenario-split → jsonl
- [x] `src/train.py` — LoRA SFT (unsloth|transformers), bf16, completion-only loss
- [x] `src/evaluate.py` — SFT vs baselines on frozen test set → metrics.json
- [x] `scripts/run_all.sh` — end-to-end
- [x] `README.md`, `VERIFICATION.md`

## Verify (locally, CPU-only where possible)
- [x] `parse_output` unit checks match file `output_binary` (120/120 on agent_hard)
- [x] `prepare_data` prints 319 usable, all-slice counts, no scenario-group overlap (random/stakes/bpv)

## Code review (done)
- [x] code-reviewer run. Fixed: right-truncation dropping answer+eos; 4-bit `prepare_model_for_kbit_training`;
      scenario_group difficulty-agnostic split (agent easy/hard title leakage); best_per_vignette alignment assert.

## Known data facts (from inspection)
- Sheets named `"{model} {approach}"`; approaches: `Rule Based`, `VB`, `RRC`, `No thinking`.
- Columns differ by domain: agent has `title`/`high.stakes`/`response_text`; development has `money`/`action`, no `response_text`.
- `reasoning` + `output` (YES/NO) already parsed and clean. `accuracy == (output_binary == contractualist.prediction)`.
- `contractualist.prediction` constant per file: easy→0(NO), hard→1(YES). Scenario dup: agent 40 titles×3, dev 13 actions×5.
- DeepSeek-R1 RRC accuracy==1 counts: agent_easy 119, agent_hard 92, dev_easy 65, dev_hard 43 = 319.

## Review
- [ ] Spawn code-reviewer on the changes before declaring done.
