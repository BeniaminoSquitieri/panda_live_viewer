#!/usr/bin/env bash
# Start the VLM ROS node, pointing inference at the persistent model server.
# The model server must already be running (start_model_server.sh).

set -eo pipefail

LEROBOT_SETUP="/home/bsquitieri-iit.local/lerobot/install/setup.bash"
SOCKET_PATH="/tmp/vlm_server.sock"
VERIFIER_LOG="/home/bsquitieri/lerobot/generated_bt/experiments/verifier_events.jsonl"

# shellcheck disable=SC1090
source "$LEROBOT_SETUP"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for the model server socket to appear before starting the node
for i in $(seq 1 30); do
    if [ -S "$SOCKET_PATH" ]; then
        echo "[ros_node] Model server is ready."
        break
    fi
    echo "[ros_node] Waiting for model server socket $SOCKET_PATH ($i/30)..."
    sleep 2
done

if [ ! -S "$SOCKET_PATH" ]; then
    echo "[ros_node] ERROR: model server socket not found. Run start_model_server.sh first."
    exit 1
fi

cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" 

exec python3 -u -m vlm_live.cli \
    --ros-args \
    -p model_server_url:="$SOCKET_PATH" \
