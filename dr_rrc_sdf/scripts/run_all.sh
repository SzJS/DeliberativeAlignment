#!/usr/bin/env bash
# End-to-end pipeline: generate SDF corpus -> continued-pretrain -> merge -> generate SFT data
#                      -> SFT -> merge -> eval.
#
#   bash scripts/run_all.sh                          # the sdf_sft arm, everything not stamped
#   bash scripts/run_all.sh --arm sft_only           # the ABLATION arm (SFT from base, no SDF)
#   bash scripts/run_all.sh --from sdf_train         # resume after a crash
#   bash scripts/run_all.sh --only eval
#   bash scripts/run_all.sh --set sdf_train.epochs=1 # override any config key (dotted path)
#   bash scripts/run_all.sh --dry-run                # print the plan, run nothing
#   bash scripts/run_all.sh --force                  # ignore stamps
#   bash scripts/run_all.sh --train-venv .venv-unsloth --set model.backend=unsloth
#                                                    # training in the unsloth venv, eval in .venv
#
# TRAINING BOTH ARMS is two invocations, not one:
#   bash scripts/run_all.sh                    # sdf_sft:  base -> SDF -> SFT
#   bash scripts/run_all.sh --arm sft_only     # sft_only: base -> SFT   (SDF stages auto-skipped)
# `--arm` sets sft_train.init_from AND scopes the SFT stamps and output paths, so the second
# invocation neither collides with nor overwrites the first.
#
# Assumes deps are installed:  uv sync --extra gpu [--extra vllm]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh

# NO `export PYTHONPATH` HERE, deliberately. dr_rrc_replication needs one because it ships no
# __init__.py and imports an absolute package that isn't installed. This experiment installs
# dr_rrc_sdf editable (src-layout, hatchling), so imports resolve from any cwd and any launcher —
# including `inspect eval`, which loads task files by path in its own process and would never see
# a PYTHONPATH set by this script. If you find yourself adding one, `uv sync` is the actual fix.

CONFIG="config.yaml"
ARM="sdf_sft"
TRAIN_VENV=""
SETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="$2"; shift 2 ;;
    --arm)        ARM="$2"; shift 2 ;;
    --train-venv) TRAIN_VENV="$2"; shift 2 ;;
    --set)     SETS+=(--set "$2"); shift 2 ;;
    --only)    ONLY+=("$2"); shift 2 ;;
    --skip)    SKIP+=("$2"); shift 2 ;;
    --from)    FROM="$2"; shift 2 ;;
    --force)   FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    # 2..21 is the usage comment block; keep this range in sync if the header grows.
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$ARM" in
  sdf_sft)  INIT_FROM=sdf ;;
  sft_only) INIT_FROM=base
            # No SDF checkpoint is involved, so the SDF stages are not merely skippable — running
            # them would be wasted money and a misleading stamp.
            SKIP+=(sdf_gen sdf_train merge_sdf) ;;
  *) echo "unknown arm: $ARM (expected sdf_sft | sft_only)" >&2; exit 2 ;;
esac
SETS+=(--set "sft_train.init_from=$INIT_FROM")
STAMP_SUFFIX=".$ARM"          # scopes sft_train / merge_sft / eval stamps, consumed by lib.sh

validate_stage_names
resolve_config "$CONFIG"

trap 'echo "[ERR] run_all.sh failed at line $LINENO"' ERR

# Preflight's secrets and spec checks only matter if a GENERATION stage will actually run —
# otherwise a `--only eval` or a crash-resume on a GPU pod is blocked on an API key it never uses.
if [[ $DRY_RUN -eq 0 ]]; then
  PREFLIGHT_ARGS=("$CONFIG")
  if ! _selected sdf_gen && ! _selected sft_gen; then
    PREFLIGHT_ARGS+=(--no-generation)
  fi
  bash scripts/preflight.sh "${PREFLIGHT_ARGS[@]}"
fi

C=(--config "$CONFIG" "${SETS[@]}")

# Generation and eval run in the default venv. TRAINING may run in a DIFFERENT venv, because
# unsloth and vllm cannot coexist (unsloth caps transformers<=5.5.0, vllm needs >=5.5.3 — see
# pyproject.toml). Splitting them by environment is what lets us have both: unsloth's LoRA
# kernels for training, vLLM for eval serving, neither compromised.
#   uv sync --extra gpu --extra vllm
#   UV_PROJECT_ENVIRONMENT=.venv-unsloth uv sync --extra gpu-unsloth
#   bash scripts/run_all.sh --train-venv .venv-unsloth --set model.backend=unsloth
UV=(uv run python)
if [[ -n "$TRAIN_VENV" ]]; then
  [[ -d "$TRAIN_VENV" ]] || { echo "--train-venv $TRAIN_VENV does not exist" >&2; exit 2; }
  TRAIN=(env "UV_PROJECT_ENVIRONMENT=$TRAIN_VENV" uv run python)
  echo "[venv]  training stages -> $TRAIN_VENV ; generation/eval -> .venv"
else
  TRAIN=("${UV[@]}")
fi

run_stage sdf_gen    "${UV[@]}"    -m dr_rrc_sdf.generation.sdf_pipeline "${C[@]}"
run_stage sdf_train  "${TRAIN[@]}" -m dr_rrc_sdf.training.train_sdf      "${C[@]}"
# Merge is a first-class stage, not an afterthought: SFT must initialise from MERGED weights.
# Stacking a second LoRA on an unmerged first adapter silently trains against the base model —
# a clean run and a null SDF result that is actually a plumbing bug. See training/merge.py.
run_stage merge_sdf  "${TRAIN[@]}" -m dr_rrc_sdf.training.merge --stage sdf "${C[@]}"
run_stage sft_gen    "${UV[@]}"    -m dr_rrc_sdf.generation.sft_pipeline "${C[@]}"
run_stage sft_train  "${TRAIN[@]}" -m dr_rrc_sdf.training.train_sft      "${C[@]}"
run_stage merge_sft  "${TRAIN[@]}" -m dr_rrc_sdf.training.merge --stage sft --arm "$ARM" "${C[@]}"
run_stage eval       "${UV[@]}"    evals/run_evals.py                    "${C[@]}"

echo
echo "Done ($ARM). See artifacts/eval/report.md"
echo "Read it with README.md's 'How to read the results' — rrc_decision is NOT the headline metric."
