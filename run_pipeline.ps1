# run_pipeline.ps1
# Centralized automation orchestrator script for the Nancy Grace Roman Data Pipeline

$ErrorActionPreference = "Stop"
Clear-Host

Write-Output "====================================================================="
Write-Output "🔭 NANCY GRACE ROMAN SPACE TELESCOPE PIPELINE DEPLOYMENT ENSEMBLE"
Write-Output "====================================================================="

# 1. Verify Virtual Environment Status
if (-not (Test-Path ".venv")) {
    Write-Error "❌ [DEPLOY ERROR] Virtual environment folder (.venv) missing at project root."
}

Write-Output "🔄 [1/4] Activating python environment and clearing historical data cache..."
# Force standard environment path tracking
$env:PYTHONPATH = "src"

# Clean out old raw ASDF images and compiled binary structures to guarantee a fresh pipeline run
if (Test-Path "data/raw") { Remove-Item "data/raw/*.asdf" -ErrorAction SilentlyContinue }
if (Test-Path "models/classifier.joblib") { Remove-Item "models/classifier.joblib" -ErrorAction SilentlyContinue }

Write-Output "🛰️ [2/4] Booting primary multi-epoch 2D raw asset generator..."
# This boots main.py to create your fresh 100-frame image stack and run initial ingestion maps
& .venv/Scripts/python.exe -m roman_exoplanet_detector.main

Write-Output "🏋️ [3/4] Initializing asynchronous background ML retraining daemon task..."
# Start the machine learning worker in a completely minimized background job process, 
# ensuring your active shell doesn't lock up or drop thread counts
Start-Process -FilePath ".venv/Scripts/python.exe" -ArgumentList "-m roman_exoplanet_detector.machine_learning.retrain_worker" -NoNewWindow -Background

Write-Output "🖥️ [4/4] Launching authenticated astronomer review portal web server..."
Write-Output "---------------------------------------------------------------------"
Write-Output "🚀 Pipeline boot sequence completed successfully! Press Ctrl+C to terminate services."
Write-Output "---------------------------------------------------------------------"

# Launch your frontend Streamlit server directly into the active terminal loop
& .venv/Scripts/streamlit.exe run src/roman_exoplanet_detector/review/dashboard.py
