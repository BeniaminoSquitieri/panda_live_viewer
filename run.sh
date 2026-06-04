#!/usr/bin/env bash
# Single entry point: starts the model server if not already running,
# then starts the ROS node pointing to it.
# Usage: ./run.sh [extra ros args...]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_SETUP="/home/bsquitieri-iit.local/lerobot/install/setup.bash"

# Model is overridable so we can test the pipeline with a smaller model:
#   MODEL_PATH=/path/to/small-model ./run.sh
MODEL_PATH="${MODEL_PATH:-/home/bsquitieri-iit.local/models/Qwen3-VL-32B-Instruct}"

# Runtime state (pid/log/socket) is kept in a stable, inspectable dir inside
# the project instead of only /tmp.
RUNTIME_DIR="${RUNTIME_DIR:-$SCRIPT_DIR/runtime}"
mkdir -p "$RUNTIME_DIR"
SOCKET_PATH="${SOCKET_PATH:-$RUNTIME_DIR/vlm_server.sock}"
PID_FILE="$RUNTIME_DIR/vlm_server.pid"
SERVER_LOG="$RUNTIME_DIR/vlm_server.log"
META_FILE="$RUNTIME_DIR/vlm_server.json"
VERIFIER_LOG="/home/bsquitieri/lerobot/generated_bt/experiments/verifier_events.jsonl"

# shellcheck disable=SC1090
source "$LEROBOT_SETUP"

# ROS middleware settings (CycloneDDS). Centralised in ros_env.sh so every entry
# point uses the same values and you never have to export them by hand; it
# honours anything already set in the environment.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/ros_env.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Minimum free VRAM (MiB) needed to load the model fully on a single GPU.
# Below this the model offloads layers to CPU and inference becomes extremely
# slow (minutes per request), which makes the client time out.
# Override for smaller models, e.g. MIN_FREE_MIB=9000 for the 4B model.
MIN_FREE_MIB="${MIN_FREE_MIB:-34000}"

# GPUs to never use (e.g. reserved/unstable). Space-separated indices.
EXCLUDED_GPUS="1"

# Pick the GPU with the most free memory at startup time, skipping excluded ones.
_grep_excl=$(echo "$EXCLUDED_GPUS" | tr ' ' '|')
read -r BEST_GPU BEST_FREE < <(nvidia-smi --query-gpu=index,memory.free \
    --format=csv,noheader,nounits \
    | grep -vE "^[[:space:]]*($_grep_excl)[[:space:]]*," \
    | sort -t',' -k2 -rn | head -1 | tr ',' ' ')
export CUDA_VISIBLE_DEVICES="$BEST_GPU"
echo "[run.sh] Using GPU $BEST_GPU (${BEST_FREE} MiB free, excluding GPU(s): $EXCLUDED_GPUS)."
if [ "${BEST_FREE:-0}" -lt "$MIN_FREE_MIB" ]; then
    echo "[run.sh] WARNING: no GPU has >= ${MIN_FREE_MIB} MiB free."
    echo "[run.sh] WARNING: the model will offload to CPU and inference will be"
    echo "[run.sh] WARNING: very slow (minutes/request). Free a GPU or wait for"
    echo "[run.sh] WARNING: other jobs to finish for full-speed inference."
fi
cd "$SCRIPT_DIR"

# Kill any stale server process whose socket no longer exists
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ! kill -0 "$OLD_PID" 2>/dev/null; then
        # Process is dead — remove stale socket and pid file so we restart fresh
        echo "[run.sh] Model server (PID $OLD_PID) is dead. Cleaning up stale files..."
        rm -f "$SOCKET_PATH" "$PID_FILE"
    elif [ ! -S "$SOCKET_PATH" ]; then
        # Process alive but socket missing — kill and restart
        echo "[run.sh] Killing stale model server (PID $OLD_PID, socket missing)..."
        kill "$OLD_PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        sleep 2
    fi
fi

# If no live server (socket missing), kill any of OUR leftover model_server
# processes so zombies don't pile up and waste GPU memory.
if [ ! -S "$SOCKET_PATH" ]; then
    ZOMBIES=$(pgrep -u "$(id -u)" -f "vlm_live.model_server" || true)
    if [ -n "$ZOMBIES" ]; then
        echo "[run.sh] Cleaning up leftover model_server processes: $ZOMBIES"
        kill $ZOMBIES 2>/dev/null || true
        sleep 2
    fi
fi

# Start model server in background if the socket doesn't exist yet
if [ ! -S "$SOCKET_PATH" ]; then
    echo "[run.sh] Starting model server in background (log: $SERVER_LOG)..."
    nohup python3 -u -m vlm_live.model_server \
        --model-path "$MODEL_PATH" \
        --socket-path "$SOCKET_PATH" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"
    # Write metadata for monitoring (status.sh reads this).
    cat > "$META_FILE" <<EOF
{
  "pid": $SERVER_PID,
  "model_path": "$MODEL_PATH",
  "gpu": "$BEST_GPU",
  "gpu_free_mib_at_start": "$BEST_FREE",
  "socket_path": "$SOCKET_PATH",
  "log": "$SERVER_LOG",
  "started_at": "$(date -Is)"
}
EOF
    echo "[run.sh] Model server PID: $SERVER_PID"
else
    echo "[run.sh] Model server already running (socket exists)."
fi

# Wait for server to be ready (socket appears once model is loaded)
echo "[run.sh] Waiting for model server to load..."
for i in $(seq 1 60); do
    if [ -S "$SOCKET_PATH" ]; then
        echo "[run.sh] Model server ready."
        break
    fi
    sleep 3
done

if [ ! -S "$SOCKET_PATH" ]; then
    echo "[run.sh] ERROR: model server did not start in time. Check $SERVER_LOG"
    exit 1
fi

# Start ROS node (foreground, Ctrl+C to stop — does NOT kill the model server)
echo "[run.sh] Starting ROS node..."
exec python3 -u -m vlm_live.cli \
    --ros-args \
    -p model_server_url:="$SOCKET_PATH" \
    -p lazy_load_model:=true \
    -p planner_dry_run:=false \
    -p require_generate_plan_service:=true \
    -p verifier_experiment_log_path:="$VERIFIER_LOG" \
    "$@"
