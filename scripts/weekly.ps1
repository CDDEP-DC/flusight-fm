# Weekly run for Windows. Usage: scripts\weekly.ps1 [-Submit] [-EnvPath <python env>]
# Without -Submit it builds the forecast and review report only.
# Uses .venv in the repo if present, otherwise the conda env "flusight" (~\.conda\envs\flusight).
param([switch]$Submit, [string]$EnvPath = "")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)
if (-not $EnvPath) {
    $EnvPath = if (Test-Path .venv) { (Resolve-Path .venv).Path } else { "$env:USERPROFILE\.conda\envs\flusight" }
}
# Put the env's python, scripts and (for conda) git/gh on PATH; works without `conda activate`.
$env:PATH = "$EnvPath;$EnvPath\Scripts;$EnvPath\Library\bin;$EnvPath\Library\cmd;$env:PATH"
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\$(Get-Date -Format yyyy-MM-dd).log"
git pull --ff-only *>> $log
flusight forecast *>> $log
if ($Submit -and $LASTEXITCODE -eq 0) { flusight submit --yes *>> $log }
