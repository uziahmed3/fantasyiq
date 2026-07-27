<#
.SYNOPSIS
  Run FantasyIQ with Python only — no Docker, no Postgres, no Redis, no Node.

.DESCRIPTION
  Same application code as the Docker stack, with two substitutions:
    Postgres -> SQLite file (fantasyiq.db)
    Redis    -> in-process TTL cache
  Both are selected by configuration, not by a code fork. The UI is a zero-build
  dashboard served by the API at /app.

  Everything lands in .venv and fantasyiq.db in this folder. Delete both to reset.

.EXAMPLE
  .\run-local.ps1                  # setup, load data, train, serve
  .\run-local.ps1 -Seasons 2024    # one season (faster)
  .\run-local.ps1 -DataUrls        # print files to download by hand
  .\run-local.ps1 -Offline         # use .\data\manual instead of downloading
  .\run-local.ps1 -Demo            # synthetic data, zero network, ~2 min
  .\run-local.ps1 -ServeOnly       # skip setup/ingest/train, just start the servers
  .\run-local.ps1 -Reset           # delete the database and start clean
#>
param(
  [string]$Seasons = "2024,2025",
  [switch]$Offline,
  [switch]$DataUrls,
  [switch]$Demo,
  [switch]$SkipTrain,
  [switch]$ServeOnly,
  [switch]$Reset,
  [int]$ApiPort = 8000,
  [int]$MlPort = 9000
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "    $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

$Root    = $PSScriptRoot
$Venv    = Join-Path $Root ".venv"
$Py      = Join-Path $Venv "Scripts\python.exe"
$DbFile  = Join-Path $Root "fantasyiq.db"
$Models  = Join-Path $Root ".models"
$Manual  = Join-Path $Root "data\manual"

# SQLAlchemy wants forward slashes; sqlite+pysqlite:///C:/path/to/file.db
$DbUrl = "sqlite+pysqlite:///" + ($DbFile -replace '\\', '/')

# Shared config for every child process. Selecting SQLite + the memory cache is
# entirely a matter of these two variables.
function Set-Env {
  $env:DATABASE_URL_OVERRIDE = $DbUrl
  $env:REDIS_URL             = "memory://"
  $env:MODEL_DIR             = $Models
  $env:MANUAL_DATA_DIR       = $Manual
  $env:ML_SERVICE_URL        = "http://127.0.0.1:$MlPort"
  $env:ENVIRONMENT           = "local"
  $env:INGEST_SEASONS        = $Seasons
  $env:PIPELINE_DATA_DIR     = Join-Path $Root ".pipeline-data"
  $env:PYTHONUNBUFFERED      = "1"
  if (-not $env:JWT_SECRET_KEY) {
    $env:JWT_SECRET_KEY = -join ((1..64) | ForEach-Object { "0123456789abcdef"[(Get-Random -Max 16)] })
  }
  if (-not $env:ACTIVE_MODEL_VERSION) { $env:ACTIVE_MODEL_VERSION = "xgboost_v1" }
}

if ($Reset) {
  Step "Resetting local state"
  Remove-Item $DbFile, (Join-Path $Root ".pipeline-data") -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item $Models -Recurse -Force -ErrorAction SilentlyContinue
  Ok "Database, models and cached raw data removed. Run again to rebuild."
  exit 0
}

# ---------------------------------------------------------------- 1. Python
Step "Checking Python"
$sys = $null
foreach ($c in @("py -3.12", "py -3.11", "python", "python3")) {
  $parts = $c.Split(" ")
  try {
    $v = & $parts[0] $parts[1..($parts.Length - 1)] --version 2>&1
    if ($LASTEXITCODE -eq 0 -and "$v" -match "Python 3\.(1[0-9]|[9])") { $sys = $c; break }
  } catch { }
}
if (-not $sys) {
  Die @"
No Python 3.9+ found on PATH.

Install from https://www.python.org/downloads/ — it does NOT require admin rights.
Tick "Add python.exe to PATH" in the installer, then open a NEW PowerShell window.
"@
}
Ok "Using: $sys  ($(& $sys.Split(' ')[0] $sys.Split(' ')[1..9] --version 2>&1))"

# ---------------------------------------------------------------- 2. venv + deps
if (-not (Test-Path $Py)) {
  Step "Creating virtual environment (.venv)"
  $p = $sys.Split(" ")
  & $p[0] $p[1..($p.Length - 1)] -m venv $Venv
  if (-not (Test-Path $Py)) { Die "venv creation failed." }
  Ok "Created."
}

$stamp = Join-Path $Venv ".deps-installed"
if (-not (Test-Path $stamp) -and -not $ServeOnly) {
  Step "Installing dependencies (3-6 minutes on first run)"
  & $Py -m pip install --upgrade pip --quiet
  # CPU-only torch: ~200MB instead of ~2.5GB of CUDA that would never be used.
  & $Py -m pip install --quiet `
      --extra-index-url https://download.pytorch.org/whl/cpu `
      -r (Join-Path $Root "backend\requirements.txt") `
      -r (Join-Path $Root "ml-service\requirements.txt") `
      -r (Join-Path $Root "pipeline\requirements.txt")
  if ($LASTEXITCODE -ne 0) {
    Die @"
pip install failed.

On a corporate network this is usually the proxy. Try:
  `$env:PIP_INDEX_URL = "<your internal PyPI mirror>"
or ask IT for the pip proxy settings, then re-run.
"@
  }
  New-Item -ItemType File -Path $stamp -Force | Out-Null
  Ok "Dependencies installed."
} elseif (-not $ServeOnly) {
  Ok "Dependencies already installed (delete .venv to force a reinstall)."
}

Set-Env
New-Item -ItemType Directory -Force -Path $Models, $Manual, $env:PIPELINE_DATA_DIR | Out-Null

# ---------------------------------------------------------------- -DataUrls
if ($DataUrls) {
  Push-Location (Join-Path $Root "pipeline")
  & $Py -m ingest --urls
  Pop-Location
  Write-Host "Save those files into: $Manual" -ForegroundColor Cyan
  exit 0
}

# ---------------------------------------------------------------- 3. schema
if (-not $ServeOnly) {
  Step "Applying database migrations to $([IO.Path]::GetFileName($DbFile))"
  Push-Location (Join-Path $Root "backend")
  & $Py -m alembic upgrade head
  $rc = $LASTEXITCODE
  Pop-Location
  if ($rc -ne 0) { Die "Migrations failed." }
  Ok "Schema ready."
}

# ---------------------------------------------------------------- 4. data
if (-not $ServeOnly) {
  Push-Location (Join-Path $Root "pipeline")
  if ($Demo) {
    Step "Loading synthetic demo data (no network)"
    & $Py -m seed_demo
    if ($LASTEXITCODE -ne 0) { Pop-Location; Die "Demo seed failed." }
  } else {
    $src = if ($Offline) { "manual" } else { "auto" }
    Step "Loading NFL data (seasons: $Seasons, source: $src)"
    if (-not $Offline) { Warn "First run downloads a few hundred MB. 2-5 minutes." }
    & $Py -m run_weekly --skip-score --source $src
    if ($LASTEXITCODE -ne 0) {
      Pop-Location
      Die @"
Could not load NFL data.

Your network is probably blocking the download. Three ways forward:

  1) Offline:  .\run-local.ps1 -DataUrls        <- lists the files to download
               save them into .\data\manual\ using your browser
               .\run-local.ps1 -Offline

  2) Hotspot:  run this once on your phone's hotspot; the data persists in
               fantasyiq.db afterwards

  3) Demo:     .\run-local.ps1 -Demo            <- synthetic data, works right now
"@
    }
  }
  Pop-Location
  Ok "Data loaded."
}

# ---------------------------------------------------------------- 5. train
if (-not $ServeOnly -and -not $SkipTrain) {
  Step "Training models on the loaded data"
  Push-Location (Join-Path $Root "ml-service")
  & $Py -m train.train_baseline
  & $Py -m train.train_xgboost
  & $Py -m train.train_torch
  Write-Host ""
  Step "Model comparison — this is the real accuracy on your data"
  & $Py -m train.evaluate
  Pop-Location
}

# ---------------------------------------------------------------- 6. projections
if (-not $ServeOnly) {
  Step "Generating projections for the upcoming week"
  Push-Location (Join-Path $Root "pipeline")
  # Needs the ML service up, so start it briefly just for the batch scoring pass.
  $tmpMl = Start-Process -FilePath $Py -PassThru -WindowStyle Hidden `
    -WorkingDirectory (Join-Path $Root "ml-service") `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$MlPort", "--log-level", "warning"
  Start-Sleep -Seconds 6
  & $Py -m run_weekly --score-only
  if ($LASTEXITCODE -ne 0) { Warn "Projection pass failed; on-demand predictions still work." }
  Stop-Process -Id $tmpMl.Id -Force -ErrorAction SilentlyContinue
  Pop-Location
}

# ---------------------------------------------------------------- 7. serve
Step "Starting services"
$ml = Start-Process -FilePath $Py -PassThru -WindowStyle Hidden `
  -WorkingDirectory (Join-Path $Root "ml-service") `
  -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$MlPort"
$api = Start-Process -FilePath $Py -PassThru -WindowStyle Hidden `
  -WorkingDirectory (Join-Path $Root "backend") `
  -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort"

$up = $false
foreach ($i in 1..60) {
  try {
    if ((Invoke-WebRequest "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200) {
      $up = $true; Ok "API healthy after ${i}s."; break
    }
  } catch { Start-Sleep -Seconds 1 }
}
if (-not $up) {
  Stop-Process -Id $ml.Id, $api.Id -Force -ErrorAction SilentlyContinue
  Die "API failed to start. Re-run with -ServeOnly after checking the error above."
}

try {
  $info = Invoke-RestMethod "http://127.0.0.1:$ApiPort/info" -TimeoutSec 5
  Ok "database=$($info.database)  cache=$($info.cache_backend)  model=$($info.active_model_version)"
} catch { }

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host " FantasyIQ is running (no Docker)" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard   http://localhost:$ApiPort/app/"
Write-Host "  API docs    http://localhost:$ApiPort/docs"
Write-Host "  ML docs     http://localhost:$MlPort/docs"
Write-Host "  Metrics     http://localhost:$ApiPort/metrics"
Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

Start-Process "http://localhost:$ApiPort/app/"

try {
  while ($true) {
    Start-Sleep -Seconds 2
    if ($api.HasExited) { Warn "API process exited."; break }
  }
} finally {
  Write-Host "`nStopping..." -ForegroundColor DarkGray
  Stop-Process -Id $ml.Id, $api.Id -Force -ErrorAction SilentlyContinue
}
