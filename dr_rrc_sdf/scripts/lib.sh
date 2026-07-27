#!/usr/bin/env bash
# Stage runner shared by run_all.sh. Source it, don't execute it.
#
# Two-layer resumability. These stamps are the COARSE layer — they skip a whole completed stage.
# The layer that actually saves a run is each stage's own mechanism:
#   sdf_gen / sft_gen  -> content-addressed response cache in outputs/cache/
#   sdf_train/sft_train-> resume_from_checkpoint
#   eval               -> inspect eval_set(log_dir=...)
# A stamp says "don't redo this"; the internal mechanism says "don't redo the 97% of this that
# already succeeded".

# NOTE: these duplicate paths.artifacts / paths.outputs in config.yaml. Shell can't read the
# config without invoking python per call, so they are hardcoded here — which means CHANGING
# paths.artifacts in config.yaml silently breaks stamping. If you ever need those paths to move,
# make run_all.sh export them from the resolved config rather than editing one side.
STAMP_DIR="artifacts/.stamps"
LOG_DIR="outputs/logs"

ONLY=(); SKIP=(); FROM=""; FORCE=0; DRY_RUN=0
STAGES=(sdf_gen sdf_train merge_sdf sft_gen sft_train merge_sft eval)

# Stages whose artifacts are arm-specific, so their stamps must be too — otherwise the second
# arm's invocation sees the first arm's stamp and skips the work.
ARM_SCOPED=(sft_train merge_sft eval)
STAMP_SUFFIX=""   # set by run_all.sh to ".<arm>"

_in_list() { local needle="$1"; shift; local x; for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done; return 1; }

# Reject unknown stage names. Without this, `--from sdf_trian` silently runs EVERYTHING and
# `--only sdf_trian` silently runs NOTHING and exits 0 — the same class of silent-typo failure
# that set_by_path exists to prevent on the config side.
validate_stage_names() {
  local name bad=0
  for name in "${ONLY[@]:-}" "${SKIP[@]:-}" ${FROM:+"$FROM"}; do
    [[ -z "$name" ]] && continue
    if ! _in_list "$name" "${STAGES[@]}"; then
      echo "unknown stage: $name" >&2; bad=1
    fi
  done
  if ((bad)); then
    echo "valid stages: ${STAGES[*]}" >&2; exit 2
  fi
}

# Should this stage run at all, given --only/--skip/--from?
_selected() {
  local name="$1"
  if ((${#ONLY[@]})); then _in_list "$name" "${ONLY[@]}" || return 1; fi
  if ((${#SKIP[@]})) && _in_list "$name" "${SKIP[@]}"; then return 1; fi
  if [[ -n "$FROM" ]]; then
    local i from_i=-1 this_i=-1
    for i in "${!STAGES[@]}"; do
      [[ "${STAGES[$i]}" == "$FROM" ]] && from_i=$i
      [[ "${STAGES[$i]}" == "$name" ]] && this_i=$i
    done
    ((from_i >= 0 && this_i >= 0 && this_i < from_i)) && return 1
  fi
  return 0
}

# Materialise artifacts/config.resolved.yaml BEFORE any stage runs.
#
# Without this, the first stage of a fresh run stamps config_sha=unresolved (the snapshot is
# written by that stage's own load_config, i.e. after run_stage has already read the sha), and
# the NEXT invocation then sees stamped=unresolved vs current=<real>, declares a config change,
# and refuses — killing the resume path on its very first use.
resolve_config() {
  local config="$1"
  [[ $DRY_RUN -eq 1 ]] && return 0
  uv run python -c "
from dr_rrc_sdf.config_utils import load_config
load_config('$config')
" >/dev/null || { echo "[FAIL] could not resolve $config" >&2; exit 1; }
}

_config_sha() {
  # READ the sha config_utils computed; do NOT hash the file. Hashing the snapshot would produce
  # a second, different "config sha" — the snapshot contains the _resolved_sha256 key, so
  # sha256sum(file) != config_sha256(cfg) — and two values under one name is how stamps and
  # manifests end up silently incomparable.
  if [[ -f artifacts/config.resolved.yaml ]]; then
    grep -m1 '^_resolved_sha256:' artifacts/config.resolved.yaml | awk '{print $2}' || echo "unresolved"
  else
    echo "unresolved"
  fi
}

# run_stage NAME CMD...
run_stage() {
  local name="$1"; shift
  _selected "$name" || { echo "[skip]  $name (deselected)"; return 0; }

  # Check --dry-run BEFORE the stamp logic, so "print the plan, run nothing" never aborts.
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[plan]  $name: $*"
    return 0
  fi

  local stamp="$STAMP_DIR/$name.done"
  if _in_list "$name" "${ARM_SCOPED[@]}"; then
    stamp="$STAMP_DIR/$name${STAMP_SUFFIX}.done"
  fi
  local cfg_sha; cfg_sha="$(_config_sha)"

  if [[ -f "$stamp" && $FORCE -eq 0 ]]; then
    # `|| true`: a stamp with no config_sha= line makes grep exit 1, which under `set -o
    # pipefail` would abort the whole script with no message.
    local stamped_sha; stamped_sha="$(grep -m1 '^config_sha=' "$stamp" 2>/dev/null | cut -d= -f2 || true)"
    if [[ -n "$stamped_sha" && "$stamped_sha" != "unresolved" && "$stamped_sha" != "$cfg_sha" ]]; then
      # dr_rrc_replication warns about this in prose ("clear artifacts/trainer when switching
      # smoke<->real"). Here it is mechanical: a stage completed under a different config is a
      # refusal, not a warning, because the resulting artifacts are silently mismatched.
      echo "[STOP]  $name was completed under a DIFFERENT config."
      echo "        stamped: $stamped_sha"
      echo "        current: $cfg_sha"
      echo "        Re-run with --force to overwrite, or clear artifacts/ if switching smoke<->real."
      return 1
    fi
    echo "[skip]  $name (already done, config unchanged)"
    return 0
  fi

  mkdir -p "$STAMP_DIR" "$LOG_DIR"
  local ts; ts="$(date +%Y%m%d-%H%M%S)"
  local log="$LOG_DIR/$name.$ts.log"
  local start; start="$(date +%s)"

  echo "[run]   $name  -> $log"
  # tee: terminal AND file (CLAUDE.md "save all terminal outputs"). pipefail makes the exit
  # status the command's, not tee's.
  if ! "$@" 2>&1 | tee "$log"; then
    local dur=$(( $(date +%s) - start ))
    echo "[FAIL]  $name after ${dur}s. Log: $log"
    echo "        Resume with:  bash scripts/run_all.sh --from $name"
    return 1
  fi

  {
    echo "stage=$name"
    echo "config_sha=$cfg_sha"
    echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "finished=$(date -Is)"
    echo "duration_s=$(( $(date +%s) - start ))"
    echo "log=$log"
  } > "$stamp"
  echo "[done]  $name in $(( $(date +%s) - start ))s"
}
