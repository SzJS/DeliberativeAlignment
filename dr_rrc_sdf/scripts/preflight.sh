#!/usr/bin/env bash
# Check everything a run needs BEFORE it starts costing money or GPU hours.
# Makes ZERO API calls and downloads nothing.
#
#   bash scripts/preflight.sh [config.yaml] [--no-generation]
#
# --no-generation skips the API-key and spec checks. run_all.sh passes it when no generation
# stage is selected, so `--only eval` and crash-resumes on a GPU pod aren't blocked on a secret
# they will never use.
set -uo pipefail
cd "$(dirname "$0")/.."
CONFIG="config.yaml"
CHECK_GENERATION=1
for arg in "$@"; do
  case "$arg" in
    --no-generation) CHECK_GENERATION=0 ;;
    *) CONFIG="$arg" ;;
  esac
done
FAIL=0
warn() { echo "  [warn] $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=1; }
ok()   { echo "  [ok]   $*"; }

echo "== environment =="
if [[ -d .venv ]]; then ok ".venv present"; else bad ".venv missing — run: uv sync"; fi
if uv run python -c "import dr_rrc_sdf" 2>/dev/null; then
  ok "dr_rrc_sdf imports (package installed editable)"
else
  bad "cannot import dr_rrc_sdf — run: uv sync   (do NOT add a PYTHONPATH shim)"
fi

echo "== secrets =="
# load_env() walks up to the repo-root .env; replicate that here without importing torch.
if uv run python - "$CONFIG" "$CHECK_GENERATION" <<'PY'
import os, sys, yaml
from dr_rrc_sdf.config_utils import load_env
load_env()
cfg = yaml.safe_load(open(sys.argv[1]))
check_gen = sys.argv[2] == "1"
key = cfg["generation"]["api_key_env"]
if os.environ.get(key):
    print(f"  [ok]   {key} is set")
elif check_gen:
    print(f"  [FAIL] {key} is EMPTY. Generation cannot run.")
    print( "         The repo-root .env exists but is empty — add the key there (it is gitignored).")
    raise SystemExit(1)
else:
    print(f"  [skip] {key} unset, but no generation stage is selected")
print("  [ok]   WANDB_API_KEY set" if os.environ.get("WANDB_API_KEY")
      else "  [warn] WANDB_API_KEY unset — W&B will run offline")
PY
then :; else FAIL=1; fi

echo "== spec =="
SPEC="assets/rrc_spec.md"
if [[ $CHECK_GENERATION -eq 0 ]]; then
  echo "  [skip] no generation stage selected"
elif [[ ! -s "$SPEC" ]]; then
  bad "$SPEC is empty"
elif grep -q "PLACEHOLDER" "$SPEC"; then
  bad "$SPEC is still the placeholder — write it before generating anything (see the TODO inside)"
else
  ok "$SPEC present (sha $(sha256sum "$SPEC" | cut -c1-12))"
fi

echo "== chat template =="
if [[ -s assets/granite_chat_template.jinja ]]; then
  ok "vendored template present ($(wc -l < assets/granite_chat_template.jinja) lines)"
else
  bad "assets/granite_chat_template.jinja missing — run scripts/fetch_chat_template.py"
fi

echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | sed 's/^/  [ok]   /'
else
  warn "no nvidia-smi — CPU-only host. Generation stages are fine; training stages are not."
fi

echo "== disk =="
avail_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if (( avail_gb < 100 )); then
  warn "${avail_gb}G free. Three merged 8B bf16 checkpoints alone are ~48G."
else
  ok "${avail_gb}G free"
fi

echo
if (( FAIL )); then echo "PREFLIGHT FAILED — fix the above before running."; exit 1; fi
echo "PREFLIGHT OK"
