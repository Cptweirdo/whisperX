# Thin wrapper -> scripts/devenv.py (Windows PowerShell). See BUILDING.md.
$ErrorActionPreference = "Stop"
$py = (Get-Command python -ErrorAction SilentlyContinue) ?? (Get-Command python3)
& $py.Source (Join-Path $PSScriptRoot "devenv.py") @args
exit $LASTEXITCODE
