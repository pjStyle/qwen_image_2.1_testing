$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot '.hf-cache'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python environment missing. Run .\setup.cmd first.'
}
& $python app.py
exit $LASTEXITCODE
