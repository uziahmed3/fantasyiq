<#
.SYNOPSIS
  One command to a working FantasyIQ instance on Windows.
.DESCRIPTION
  Boots the stack, pulls real NFL data, trains the models, and generates projections.
  Windows has no `make`, so this replaces the Makefile targets.
.EXAMPLE
  .\run.ps1                 # full setup: build, ingest, train, project
  .\run.ps1 -Seasons 2025   # only pull one season (faster)
  .\run.ps1 -SkipTrain      # boot + ingest only
  .\run.ps1 -Stop           # shut everything down
  .\run.ps1 -Nuke           # shut down and delete all data volumes
#>
param(
  [string]$Seasons = "2024,2025",
  [switch]$SkipIngest,
  [switch]$SkipTrain,
  [switch]$Stop,
  [switch]$Nuke,
  [switch]$Logs
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------- teardown shortcuts ----------
if ($Stop) { docker compose down; exit 0 }
if ($Nuke) { docker compose down -v; Ok "Containers and volumes removed."; exit 0 }
if ($Logs) { docker compose logs -f backend ml-service; exit 0 }

# ---------- preflight ----------
Step "Checking Docker"
try {
  docker version --format '{{.Server.Version}}' | Out-Null
} catch {
  Die @"
Docker is not running.

Install Docker Desktop:  https://www.docker.com/products/docker-desktop/
Then start it, wait for the whale icon in the tray to stop animating, and re-run this script.
"@
}
Ok "Docker is running."

if (-not (Test-Path ".env")) {
  Step "Creating .env from .env.example"
  Copy-Item ".env.example" ".env"
  # Give the local instance a real random JWT secret instead of the placeholder.
  $secret = -join ((1..64) | ForEach-Object { "0123456789abcdef"[(Get-Random -Max 16)] })
  (Get-Content ".env") -replace '^JWT_SECRET_KEY=.*', "JWT_SECRET_KEY=$secret" |
    Set-Content ".env"
  (Get-Content ".env") -replace '^INGEST_SEASONS=.*', "INGEST_SEASONS=$Seasons" |
    Set-Content ".env"
  Ok ".env created with a random JWT secret."
} else {
  Ok ".env already exists - leaving it alone."
}

# ---------- build & boot ----------
Step "Building images (first run pulls ~2GB and takes 5-10 minutes)"
docker compose build
if ($LASTEXITCODE -ne 0) { Die "Image build failed. Run '.\run.ps1 -Logs' for detail." }

Step "Starting Postgres, Redis, ML service, API, dashboard, Grafana"
docker compose up -d
if ($LASTEXITCODE -ne 0) { Die "docker compose up failed." }

Step "Waiting for the API to become healthy"
$healthy = $false
foreach ($i in 1..90) {
  try {
    $r = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -UseBasicParsing
    if ($r.StatusCode -eq 200) { $healthy = $true; Ok "API healthy after ${i}s."; break }
  } catch { Start-Sleep -Seconds 1 }
}
if (-not $healthy) {
  docker compose logs --tail 40 backend
  Die "API did not come up. Logs above."
}

# ---------- data ----------
if (-not $SkipIngest) {
  Step "Pulling real NFL data (seasons: $Seasons) and loading Postgres"
  Warn "First run downloads a few hundred MB from nflverse; 2-5 minutes is normal."
  docker compose run --rm -e INGEST_SEASONS=$Seasons pipeline python -m run_weekly --skip-score
  if ($LASTEXITCODE -ne 0) { Die "Ingest failed. Check your internet connection and retry." }

  $count = docker compose exec -T postgres psql -U fantasyiq -d fantasyiq -tAc `
    "SELECT COUNT(*) FROM player_stats"
  Ok "player_stats rows loaded: $($count.Trim())"
}

if (-not $SkipTrain) {
  Step "Training on the real data (ridge, XGBoost, PyTorch) and comparing"
  docker compose run --rm ml-service python -m train.train_baseline
  docker compose run --rm ml-service python -m train.train_xgboost
  docker compose run --rm ml-service python -m train.train_torch
  Write-Host ""
  Step "Model bake-off - this is your real accuracy"
  docker compose run --rm ml-service python -m train.evaluate

  Step "Restarting the ML service to pick up the new artifacts"
  docker compose restart ml-service
  Start-Sleep -Seconds 8
}

if (-not $SkipIngest) {
  Step "Generating projections for the upcoming week"
  docker compose run --rm pipeline python -m run_weekly --skip-ingest
}

# ---------- done ----------
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host " FantasyIQ is running" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard    http://localhost:3000"
Write-Host "  API docs     http://localhost:8000/docs"
Write-Host "  ML docs      http://localhost:9000/docs"
Write-Host "  Grafana      http://localhost:3001   (admin / admin)"
Write-Host "  Prometheus   http://localhost:9090"
Write-Host ""
Write-Host "  Stop it      .\run.ps1 -Stop"
Write-Host "  Logs         .\run.ps1 -Logs"
Write-Host "  Wipe data    .\run.ps1 -Nuke"
Write-Host ""
