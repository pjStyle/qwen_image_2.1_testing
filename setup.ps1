$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.uv-python'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Install uv from https://docs.astral.sh/uv/getting-started/installation/'
}

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
uv pip install --python $python 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Could not install CUDA PyTorch.' }

uv pip install --python $python -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install app dependencies.' }

& $python -m qwen_workspace.check_environment
if ($LASTEXITCODE -ne 0) { throw 'The GPU environment check failed.' }

Write-Host 'Setup complete. Run .\launch.cmd to start the local interface.'
