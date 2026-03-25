# Living AI System — PowerShell Launch Script
# Usage:
#   .\launch.ps1 dev       — Start in development mode
#   .\launch.ps1 prod      — Start in production mode (Docker)
#   .\launch.ps1 stop      — Stop all services
#   .\launch.ps1 install   — Install all dependencies

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("dev", "prod", "stop", "install")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Write-Step {
    param([string]$Message)
    Write-Host "`n>>> $Message" -ForegroundColor Cyan
}

function Assert-Command {
    param([string]$Command, [string]$InstallHint)
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$Command' not found. $InstallHint" -ForegroundColor Red
        exit 1
    }
}

function Start-Dev {
    Write-Step "Starting Living AI System in DEVELOPMENT mode"

    Assert-Command "python" "Install Python 3.12+ from https://python.org"
    Assert-Command "node" "Install Node.js LTS from https://nodejs.org"

    # Check .env exists
    $EnvFile = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $EnvFile)) {
        Write-Step "Creating .env from .env.example"
        Copy-Item (Join-Path $ProjectRoot ".env.example") $EnvFile
        Write-Host "WARNING: .env created from example. Edit it before production use." -ForegroundColor Yellow
    }

    # Activate virtual environment
    $VenvPath = Join-Path $ProjectRoot ".venv"
    if (-not (Test-Path $VenvPath)) {
        Write-Step "Creating virtual environment"
        python -m venv $VenvPath
    }

    $ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    Write-Step "Activating virtual environment"
    & $ActivateScript

    # Install Python dependencies
    Write-Step "Installing Python dependencies"
    pip install -r (Join-Path $ProjectRoot "requirements.txt") --break-system-packages --quiet

    # Install frontend dependencies
    Write-Step "Installing frontend dependencies"
    Push-Location (Join-Path $ProjectRoot "frontend")
    npm install --silent
    Pop-Location

    # Create data directories
    $DataDir = Join-Path $ProjectRoot "data"
    $ModelsDir = Join-Path $ProjectRoot "models"
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

    # Start backend
    Write-Step "Starting backend on http://localhost:8000"
    $BackendJob = Start-Job -ScriptBlock {
        param($Root, $Activate)
        & $Activate
        Set-Location $Root
        python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
    } -ArgumentList $ProjectRoot, $ActivateScript

    # Start frontend
    Write-Step "Starting frontend on http://localhost:3000"
    $FrontendJob = Start-Job -ScriptBlock {
        param($FrontendRoot)
        Set-Location $FrontendRoot
        npm run dev
    } -ArgumentList (Join-Path $ProjectRoot "frontend")

    Write-Host "`n=== Living AI System is running ===" -ForegroundColor Green
    Write-Host "Backend:  http://localhost:8000" -ForegroundColor Green
    Write-Host "Frontend: http://localhost:3000" -ForegroundColor Green
    Write-Host "Health:   http://localhost:8000/health" -ForegroundColor Green
    Write-Host "`nPress Ctrl+C to stop all services`n" -ForegroundColor Yellow

    try {
        while ($true) {
            Start-Sleep -Seconds 2
            $BackendJob | Receive-Job
            $FrontendJob | Receive-Job
        }
    } finally {
        Stop-Job $BackendJob, $FrontendJob
        Remove-Job $BackendJob, $FrontendJob
        Write-Host "`nAll services stopped." -ForegroundColor Yellow
    }
}

function Start-Prod {
    Write-Step "Starting Living AI System in PRODUCTION mode"

    Assert-Command "docker" "Install Docker Desktop from https://docker.com/products/docker-desktop"
    Assert-Command "docker" "Ensure Docker Desktop is running"

    $EnvFile = Join-Path $ProjectRoot ".env"
    if (-not (Test-Path $EnvFile)) {
        Write-Host "ERROR: .env file not found. Copy .env.example to .env and configure it." -ForegroundColor Red
        exit 1
    }

    Write-Step "Building and starting all services"
    Set-Location $ProjectRoot
    docker compose up -d --build

    Write-Step "Waiting for services to become healthy"
    $MaxWait = 120
    $Elapsed = 0
    while ($Elapsed -lt $MaxWait) {
        Start-Sleep -Seconds 5
        $Elapsed += 5
        $Health = docker compose ps --format json 2>$null
        Write-Host "Waiting... ($Elapsed/$MaxWait seconds)" -ForegroundColor Gray
        try {
            $Response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 3
            if ($Response.StatusCode -eq 200) {
                break
            }
        } catch {
            # Not ready yet
        }
    }

    Write-Host "`n=== Living AI System is running in production ===" -ForegroundColor Green
    Write-Host "Backend:  http://localhost:8000" -ForegroundColor Green
    Write-Host "Frontend: http://localhost:3000" -ForegroundColor Green
    Write-Host "Health:   http://localhost:8000/health" -ForegroundColor Green
    Write-Host "`nTo view logs: docker compose logs -f" -ForegroundColor Yellow
    Write-Host "To stop:      .\launch.ps1 stop`n" -ForegroundColor Yellow
}

function Stop-All {
    Write-Step "Stopping Living AI System"
    Assert-Command "docker" "Docker not found"
    Set-Location $ProjectRoot
    docker compose down
    Write-Host "All services stopped." -ForegroundColor Green
}

function Install-All {
    Write-Step "Installing all dependencies"

    Assert-Command "python" "Install Python 3.12+ from https://python.org"
    Assert-Command "node" "Install Node.js LTS from https://nodejs.org"
    Assert-Command "pip" "pip should come with Python"

    # Python virtual environment
    $VenvPath = Join-Path $ProjectRoot ".venv"
    if (-not (Test-Path $VenvPath)) {
        python -m venv $VenvPath
    }

    $ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    & $ActivateScript

    Write-Step "Installing Python packages"
    pip install -r (Join-Path $ProjectRoot "requirements.txt") --break-system-packages

    Write-Step "Installing Playwright browsers"
    python -m playwright install chromium

    Write-Step "Installing frontend packages"
    Push-Location (Join-Path $ProjectRoot "frontend")
    npm install
    Pop-Location

    Write-Host "`nInstallation complete." -ForegroundColor Green
}

switch ($Mode) {
    "dev"     { Start-Dev }
    "prod"    { Start-Prod }
    "stop"    { Stop-All }
    "install" { Install-All }
}
