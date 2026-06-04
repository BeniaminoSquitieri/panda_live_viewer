#!/usr/bin/env bash
# Single entry point: starts the model server if not already running,
# then starts the ROS node pointing to it.
# Usage: ./run.sh [extra ros args...]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_SETUP="/home/bsquitieri-iit.local/lerobot/install/setup.bash"
MODEL_PATH="/home/bsquitieri-iit.local/models/Qwen3-VL-32B-Instruct"
SOCKET_PATH="/tmp/vlm_server.sock"
PID_FILE="/tmp/vlm_server.pid"
VERIFIER_LOG="/home/bsquitieri/lerobot/generated_bt/experiments/verifier_events.jsonl"
SERVER_LOG="/tmp/vlm_server.log"

# shellcheck disable=SC1090
source "$LEROBOT_SETUP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pick the GPU with the most free memory at startup time.
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="$BEST_GPU"
echo "[run.sh] Using GPU $BEST_GPU (most free memory)."
cd "$SCRIPT_DIR"

# Kill any stale server process whose socket no longer exists
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null && [ ! -S "$SOCKET_PATH" ]; then
        echo "[run.sh] Killing stale model server (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
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
