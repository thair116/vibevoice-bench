#!/bin/bash
#
# VibeVoice Mac Benchmark — one-command runner.
#
# What this does:
#   1. Checks your Mac is supported (Apple Silicon, Python 3.10+, enough disk).
#   2. Sets up a Python venv with the dependencies.
#   3. Starts the VibeVoice TTS server in the background (downloads ~10 GB
#      model on first run).
#   4. Runs a ~60-second benchmark across batch sizes 1, 2, 4.
#   5. Saves results to results_<hostname>_<timestamp>.json.
#   6. Shuts the server down cleanly.
#
# Usage:
#   ./run.sh
#
# Optional environment overrides:
#   MAX_BATCH_SIZE=8 ./run.sh    # if you have a lot of RAM (default: 4)
#   PORT=7070 ./run.sh           # if 6969 is already in use (default: 6969)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-6969}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
SERVER_URL="http://localhost:${PORT}"
SERVER_LOG="${SCRIPT_DIR}/server.log"
SERVER_PID=""

# ---------- helpers ----------
log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (pid $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ---------- 1. sanity checks ----------
log "Checking your Mac..."

[[ "$(uname -s)" == "Darwin" ]] || fail "This benchmark only runs on macOS. You appear to be on $(uname -s)."

if [[ "$(uname -m)" != "arm64" ]]; then
    fail "This benchmark requires an Apple Silicon Mac (M1/M2/M3/M4).
       Your machine reports architecture: $(uname -m).
       Intel Macs would fall back to CPU and take many hours per run."
fi

command -v python3 >/dev/null 2>&1 || fail "python3 is not installed. Install it from https://www.python.org/downloads/ (3.10 or newer)."

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if (( PY_MAJOR < 3 )) || { (( PY_MAJOR == 3 )) && (( PY_MINOR < 10 )); }; then
    fail "Python ${PY_VER} found, but 3.10 or newer is required. Install from https://www.python.org/downloads/"
fi

DISK_AVAIL_KB=$(df -k "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
DISK_AVAIL_GB=$((DISK_AVAIL_KB / 1024 / 1024))
if (( DISK_AVAIL_GB < 15 )); then
    fail "Need at least 15 GB free disk space (you have ${DISK_AVAIL_GB} GB). Free up some space and try again."
fi

if lsof -i ":${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    fail "Port ${PORT} is already in use. Close whatever's using it, or run: PORT=7070 ./run.sh"
fi

CHIP=$(sysctl -n machdep.cpu.brand_string)
RAM_BYTES=$(sysctl -n hw.memsize)
RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
MACOS_VERSION=$(sw_vers -productVersion)
HOSTNAME_SHORT=$(scutil --get LocalHostName 2>/dev/null || hostname -s)

log "System looks good:"
echo "    Chip:    $CHIP"
echo "    RAM:     ${RAM_GB} GB"
echo "    macOS:   $MACOS_VERSION"
echo "    Python:  $PY_VER"
echo "    Disk:    ${DISK_AVAIL_GB} GB free"

# ---------- 2. python venv + deps ----------
VENV="${SCRIPT_DIR}/.venv"
if [[ ! -d "$VENV" ]]; then
    log "Creating Python virtual environment (.venv)..."
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

log "Installing Python dependencies (this can take a minute on first run)..."
pip install --upgrade pip -q
pip install -r "${SCRIPT_DIR}/server/requirements.txt" -q
pip install git+https://github.com/rsxdalv/VibeVoice.git -q

# ---------- 3. start server ----------
export VOICES_DIR="${SCRIPT_DIR}/voices"
export VIBEVOICE_MODEL_PATH="rsxdalv/VibeVoice-Large"
export VIBEVOICE_MODEL_LOCAL_PATH="${SCRIPT_DIR}/model"
export PORT
export MAX_BATCH_SIZE
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [[ ! -d "${SCRIPT_DIR}/model" ]] || [[ -z "$(ls -A "${SCRIPT_DIR}/model" 2>/dev/null)" ]]; then
    log "First-time setup: the server will download the VibeVoice-Large model
    (~10 GB) from HuggingFace before it's ready. This usually takes 10–15
    minutes on a fast connection. Subsequent runs will skip this step."
fi

log "Starting VibeVoice server on ${SERVER_URL}..."
echo "    Logs: $SERVER_LOG"

# Start server in background, redirect both streams to the log file
python3 "${SCRIPT_DIR}/server/main.py" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# ---------- 4. wait for /health ----------
log "Waiting for server to become healthy (timeout: 30 minutes)..."
HEALTHY=0
DEADLINE=$(( $(date +%s) + 1800 ))   # 30 min
while (( $(date +%s) < DEADLINE )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "--- last 60 lines of server.log ---"
        tail -n 60 "$SERVER_LOG" || true
        fail "Server process exited before becoming healthy. See $SERVER_LOG above."
    fi
    if curl -sf --max-time 3 "${SERVER_URL}/health" >/dev/null 2>&1; then
        HEALTHY=1
        echo
        break
    fi
    printf '.'
    sleep 5
done

if (( HEALTHY == 0 )); then
    echo
    echo "--- last 60 lines of server.log ---"
    tail -n 60 "$SERVER_LOG" || true
    fail "Server did not become healthy within 30 minutes. See $SERVER_LOG above."
fi

log "Server is ready."

# ---------- 5. run benchmark ----------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RAW_RESULTS="${SCRIPT_DIR}/.raw_results_${TIMESTAMP}.json"
FINAL_RESULTS="${SCRIPT_DIR}/results_${HOSTNAME_SHORT}_${TIMESTAMP}.json"

log "Running benchmark (≈60 seconds of audio, batch sizes 1, 2, 4, 3 runs each)..."
python3 "${SCRIPT_DIR}/benchmark/benchmark.py" \
    --server "$SERVER_URL" \
    --batch-sizes 1,2,4 \
    --runs 3 \
    --max-duration 60 \
    --output "$RAW_RESULTS"

# ---------- 6. enrich results with system info ----------
log "Saving final results..."
python3 - <<PYEOF
import json, sys
from pathlib import Path

raw = Path("$RAW_RESULTS")
final = Path("$FINAL_RESULTS")

with raw.open() as f:
    data = json.load(f)

data["system_info"] = {
    "chip": "$CHIP",
    "ram_gb": $RAM_GB,
    "macos_version": "$MACOS_VERSION",
    "python_version": "$PY_VER",
    "hostname": "$HOSTNAME_SHORT",
    "max_batch_size_env": "$MAX_BATCH_SIZE",
}

# Move system_info to the top for readability.
ordered = {"system_info": data["system_info"]}
for k, v in data.items():
    if k != "system_info":
        ordered[k] = v

with final.open("w") as f:
    json.dump(ordered, f, indent=2)

raw.unlink()
PYEOF

# ---------- 7. done ----------
echo
log "Done!"
echo
echo "    Results file:  $FINAL_RESULTS"
echo
echo "    Please send this file back."
echo
