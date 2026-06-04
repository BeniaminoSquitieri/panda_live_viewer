#!/usr/bin/env bash
# Monitor the VLM model server + ROS node state.
# Usage:
#   ./status.sh            one-shot status
#   ./status.sh -f         follow the server log (tail -f)
#   ./status.sh -w         watch status, refreshing every 2s

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${RUNTIME_DIR:-$SCRIPT_DIR/runtime}"
SOCKET_PATH="${SOCKET_PATH:-$RUNTIME_DIR/vlm_server.sock}"
PID_FILE="$RUNTIME_DIR/vlm_server.pid"
SERVER_LOG="$RUNTIME_DIR/vlm_server.log"
META_FILE="$RUNTIME_DIR/vlm_server.json"

print_status() {
    echo "==================== VLM SERVER STATUS ===================="
    # Metadata
    if [ -f "$META_FILE" ]; then
        echo "--- metadata ($META_FILE) ---"
        cat "$META_FILE"
        echo
    fi

    # Server process
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Model server: ALIVE (PID $PID)"
            ps -o pid,etime,%cpu,%mem,rss --no-headers -p "$PID" \
                | awk '{printf "  uptime=%s  cpu=%s%%  mem=%s%%  rss=%.1fGB\n", $2,$3,$4,$5/1024/1024}'
        else
            echo "Model server: DEAD (stale PID $PID)"
        fi
    else
        echo "Model server: NOT STARTED (no pid file)"
    fi

    # Socket
    if [ -S "$SOCKET_PATH" ]; then
        echo "Socket: present ($SOCKET_PATH)"
    else
        echo "Socket: MISSING ($SOCKET_PATH)"
    fi

    # ROS node
    if pgrep -u "$(id -u)" -f "vlm_live.cli" >/dev/null 2>&1; then
        echo "ROS node: RUNNING ($(pgrep -u "$(id -u)" -f vlm_live.cli | tr '\n' ' '))"
    else
        echo "ROS node: not running"
    fi

    # GPU usage (our processes highlighted)
    echo "--- GPU memory (MiB free / total) ---"
    nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu \
        --format=csv,noheader,nounits | awk -F',' '{printf "  GPU %s: %s free / %s total  util=%s%%\n",$1,$2,$3,$4}'

    echo "--- our GPU processes ---"
    OURPIDS=$(pgrep -u "$(id -u)" -f "vlm_live" | tr '\n' '|' | sed 's/|$//')
    if [ -n "$OURPIDS" ]; then
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
            | grep -E "^[[:space:]]*($OURPIDS)," || echo "  (none on GPU yet — may be CPU-offloaded)"
    else
        echo "  (no vlm processes)"
    fi

    # Last inference result
    if [ -f "$SERVER_LOG" ]; then
        echo "--- last log lines ---"
        tail -n 6 "$SERVER_LOG"
    fi
    echo "=========================================================="
}

case "${1:-}" in
    -f|--follow)
        exec tail -n 50 -f "$SERVER_LOG"
        ;;
    -w|--watch)
        while true; do
            clear
            print_status
            sleep 2
        done
        ;;
    *)
        print_status
        ;;
esac
