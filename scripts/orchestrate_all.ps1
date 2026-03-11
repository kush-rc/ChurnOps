# Master Orchestrator Script
# ========================
# This script automatically trains the ML pipeline and runs Optuna Tuning 
# across ALL 8 million-row datasets. 
# 
# WARNING: Training 48 different Machine Learning models (6 algorithms * 8 datasets) 
# across 8 Million rows locally will take SEVERAL HOURS. Run this overnight!

$ErrorActionPreference = "Stop"

$domains = @(
    "telco",
    "banking",
    "ecommerce",
    "ott",
    "healthcare",
    "gaming",
    "saas",
    "hospitality"
)

$pythonExe = ".\venv\Scripts\python.exe"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING MASTER MLOPS ORCHESTRATION (8 MILLION ROWS)" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# Detect Dedicated Hardware
Write-Host "`nScanning Hardware for CUDA/GPU Accelerators..." -ForegroundColor Yellow
$gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -ExpandProperty Name -First 1

if ($gpu) {
    Write-Host "✅ DETECTED HARDWARE: $gpu" -ForegroundColor Green
    Write-Host "🔥 Engaging CUDA/GPU Acceleration for XGBoost, LightGBM, and CatBoost!" -ForegroundColor Red
} else {
    Write-Host "⚠️ No NVIDIA GPU detected via WMI. Ensure drivers are installed. Defaulting to CPU..." -ForegroundColor Yellow
}

foreach ($domain in $domains) {
    Write-Host "`n======================================================" -ForegroundColor Yellow
    Write-Host "▶️ PROCESSING DOMAIN: $($domain.ToUpper())" -ForegroundColor Yellow
    Write-Host "======================================================" -ForegroundColor Yellow

    # 0. Run the Data Processing Pipeline (Raw CSV -> Features Parquet)
    Write-Host "Starting Data Pipeline (ETL & Features) for $domain..." -ForegroundColor Green
    & $pythonExe "scripts\test_pipeline.py" --dataset $domain

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error occurred during data processing for $domain. Skipping..." -ForegroundColor Red
        continue
    }

    # 1. Run the Full Training Pipeline (Train 6 Models, Compare, Register Best)
    Write-Host "`nStarting Core Training Pipeline for $domain..." -ForegroundColor Green
    & $pythonExe "scripts\run_training.py" --dataset $domain

    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error occurred during training for $domain. Skipping tuning..." -ForegroundColor Red
        continue
    }

    # 2. Run Optuna Hyperparameter Tuning
    Write-Host "`nStarting Optuna Hyperparameter Tuning for $domain..." -ForegroundColor Green
    & $pythonExe "scripts\run_tuning.py" --dataset $domain
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Error occurred during tuning for $domain." -ForegroundColor Red
    }
}

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "🎉 MASTER ORCHESTRATION COMPLETE!" -ForegroundColor Cyan
Write-Host "All models are trained, tuned, and stored in MLflow." -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
