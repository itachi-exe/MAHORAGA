#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Main server Python ────────────────────────────────────────────────────────
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Error: python3 not found. Install Python 3 first." >&2
  exit 1
fi

# ── STORM scheduler Python ────────────────────────────────────────────────────
# xgboost / lightgbm live in ~/.local; must unset VIRTUAL_ENV so python3.13
# doesn't lock itself into build_venv which is missing those packages.
STORM_LOCAL="/home/d3f4ult/.local/lib/python3.13/site-packages"
STORM_PYTHON_BIN=""
STORM_PYTHONPATH=""

for candidate in python3.13 python3 python; do
  bin="$(command -v "$candidate" 2>/dev/null || true)"
  [[ -z "$bin" ]] && continue
  if VIRTUAL_ENV="" PYTHONPATH="$STORM_LOCAL" "$bin" \
       -c "import xgboost, lightgbm, joblib" 2>/dev/null; then
    STORM_PYTHON_BIN="$bin"
    STORM_PYTHONPATH="$STORM_LOCAL"
    break
  fi
  # fallback: try without extra PYTHONPATH (already has everything)
  if VIRTUAL_ENV="" "$bin" -c "import xgboost, lightgbm, joblib" 2>/dev/null; then
    STORM_PYTHON_BIN="$bin"
    break
  fi
done

STORM_PID=""

# ── Launch STORM scheduler in background ──────────────────────────────────────
if [[ -n "$STORM_PYTHON_BIN" ]]; then
  echo "Starting STORM scheduler  →  $STORM_PYTHON_BIN"
  VIRTUAL_ENV="" PYTHONPATH="${STORM_PYTHONPATH:-}" \
    "$STORM_PYTHON_BIN" "$SCRIPT_DIR/STORM/bot/scheduler.py" \
    >> "$SCRIPT_DIR/STORM/storm_scheduler.log" 2>&1 &
  STORM_PID=$!
  echo "STORM scheduler running  PID=$STORM_PID  (log: STORM/storm_scheduler.log)"
else
  echo "WARNING: xgboost/lightgbm not found — STORM scheduler not started" >&2
fi

# ── Graceful shutdown: stop STORM when server exits ───────────────────────────
cleanup() {
  if [[ -n "$STORM_PID" ]] && kill -0 "$STORM_PID" 2>/dev/null; then
    echo "Stopping STORM scheduler (PID $STORM_PID)..."
    kill "$STORM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ── Start main server (foreground) ───────────────────────────────────────────
echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" "$SCRIPT_DIR/server.py"
