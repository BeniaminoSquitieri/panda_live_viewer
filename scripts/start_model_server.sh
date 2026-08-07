#!/usr/bin/env bash
# Start the persistent VLM model server.
# The model is loaded once into GPU RAM and stays there until this process dies.
# Run this once (e.g. at system boot via systemd) and leave it running.
# Communicates via Unix socket — no network port is opened.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(dirname "$PROJECT_DIR")}"

LEROBOT_SETUP="${LEROBOT_SETUP:-$LEROBOT_ROOT/install/setup.bash}"
MODEL_PATH="${MODEL_PATH:-${VLM_MODEL_PATH:-Qwen/Qwen3-VL-32B-Instruct}}"
SOCKET_PATH="${SOCKET_PATH:-$PROJECT_DIR/runtime/vlm_server.sock}"

if [ -f "$LEROBOT_SETUP" ]; then source "$LEROBOT_SETUP"; else echo "[model_server] ROS setup not found; continuing because the socket server is ROS-free."; fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_DIR"
echo "[model_server] Starting — model: $MODEL_PATH, socket: $SOCKET_PATH"
exec python3 -u -m vlm_live.model_server \
    --model-path "$MODEL_PATH" \
    --socket-path "$SOCKET_PATH"
