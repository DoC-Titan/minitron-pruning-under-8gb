$ErrorActionPreference = 'Stop'
$Python = Join-Path (Split-Path $PSScriptRoot -Parent) 'Minitron_study_DAY_1\.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Set $Python to your compatible environment; read REPRODUCE.md.' }
if (Test-Path (Join-Path $PSScriptRoot 'STOP')) { throw 'STOP file is present. Remove it only when ready to resume.' }
& $Python -u (Join-Path $PSScriptRoot 'run.py')
if ($LASTEXITCODE -ne 0) { throw 'Run paused or failed. Read RUN_STATUS.json and logs/runner.log. Checkpoints retained.' }
