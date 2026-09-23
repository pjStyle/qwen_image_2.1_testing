#!/usr/bin/env bash
set -euo pipefail

script_path="${BASH_SOURCE[0]}"
if [[ "$script_path" == */* ]]; then
    cd "${script_path%/*}"
fi
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"

if ! command -v git >/dev/null 2>&1; then
    echo 'Git is required. Install it in the pod image first.' >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    if ! command -v curl >/dev/null 2>&1; then
        echo 'curl is required to install uv.' >&2
        exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo 'uv installation failed.' >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi

uv pip install --python .venv/bin/python 'torch==2.11.0' 'torchvision==0.26.0' \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m qwen_workspace.check_environment

echo 'Setup complete. Set QWEN_AUTH_PASSWORD, then run: bash launch_runpod.sh'
