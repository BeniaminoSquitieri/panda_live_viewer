#!/usr/bin/env bash
# Start the persistent VLM model server.
# The model is loaded once into GPU RAM and stays there until this process dies.
# Run this once (e.g. at system boot via systemd) and leave it running.
# Communicates via Unix socket — no network port is opened.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LEROBOT_SETUP="/home/bsquitieri-iit.local/lerobot/install/setup.bash"
MODEL_PATH="/home/bsquitieri-iit.local/models/Qwen3-VL-32B-Instruct"
SOCKET_PATH="/tmp/vlm_server.sock"

# shellcheck disable=SC1090
source "$LEROBOT_SETUP"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_DIR"
echo "[model_server] Starting — model: $MODEL_PATH, socket: $SOCKET_PATH"
exec python3 -u -m vlm_live.model_server \
    --model-path "$MODEL_PATH" \
    --socket-path "$SOCKET_PATH"
