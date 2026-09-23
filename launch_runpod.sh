#!/usr/bin/env bash
set -euo pipefail

script_path="${BASH_SOURCE[0]}"
if [[ "$script_path" == */* ]]; then
    cd "${script_path%/*}"
fi
if [[ ! -x .venv/bin/python ]]; then
    echo 'Python environment missing. Run: bash setup_runpod.sh' >&2
    exit 1
fi
if [[ -z "${QWEN_AUTH_PASSWORD:-}" ]]; then
    echo 'Set QWEN_AUTH_PASSWORD before exposing the app through Runpod.' >&2
    exit 1
fi

export HF_HOME="$PWD/.hf-cache"
export QWEN_HOST=0.0.0.0
export QWEN_PORT="${QWEN_PORT:-7860}"
exec .venv/bin/python app.py
