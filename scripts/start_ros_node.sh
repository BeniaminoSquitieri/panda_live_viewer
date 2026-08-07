#!/usr/bin/env bash
# Start the VLM ROS node, pointing inference at the persistent model server.
# The model server must already be running (start_model_server.sh).

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(dirname "$PROJECT_DIR")}"
LEROBOT_SETUP="${LEROBOT_SETUP:-$LEROBOT_ROOT/install/setup.bash}"
SOCKET_PATH="${SOCKET_PATH:-$PROJECT_DIR/runtime/vlm_server.sock}"
VERIFIER_LOG="${VERIFIER_LOG:-$LEROBOT_ROOT/generated_bt/experiments/verifier_events.jsonl}"
VLM_CAMERA_VIEW="${VLM_CAMERA_VIEW:-front}"

[ -f "$LEROBOT_SETUP" ] || { echo "[ros_node] ERROR: ROS workspace setup not found: $LEROBOT_SETUP"; exit 1; }
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

cd "$PROJECT_DIR"

exec python3 -u -m vlm_live.cli --ros-args -p model_server_url:="$SOCKET_PATH" -p lazy_load_model:=true -p vlm_camera_view:="$VLM_CAMERA_VIEW" -p planner_dry_run:=false -p require_generate_plan_service:=true -p verifier_experiment_log_path:="$VERIFIER_LOG" "$@"
