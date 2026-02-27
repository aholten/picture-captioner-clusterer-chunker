#!/usr/bin/env bash
set -e

BACKEND="${1:-local}"
MODEL="${2:-Qwen/Qwen2.5-VL-3B-Instruct}"
BATCH=5000

while true; do
    uv run --extra local python caption.py run \
        --backend "$BACKEND" --model "$MODEL" \
        --restart-every "$BATCH"
    EXIT=$?
    if [ $EXIT -eq 0 ]; then
        echo "All photos captioned."
        break
    elif [ $EXIT -eq 2 ]; then
        echo "Batch of $BATCH done. Restarting..."
    else
        echo "Error (exit $EXIT). Stopping."
        exit $EXIT
    fi
done
