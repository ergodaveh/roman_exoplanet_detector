@echo off
cls
echo =====================================================================
echo NANCY GRACE ROMAN SPACE TELESCOPE PIPELINE DEPLOYMENT ENSEMBLE
echo =====================================================================

:: Point to the parent directory containing your virtual environment .venv binary folder
cd /d "C:\Firebase-app\roman-exoplanet-detector"

:: Set python path environment strings relative to your repository files folder
set PYTHONPATH=roman_exoplanet_detector\src;roman_exoplanet_detector

echo [1/4] Clearing historical telemetry data cache...
if exist roman_exoplanet_detector\data\raw del /q roman_exoplanet_detector\data\raw\*.asdf >nul 2>&1
if exist roman_exoplanet_detector\models\classifier.joblib del /q roman_exoplanet_detector\models\classifier.joblib >nul 2>&1

echo [2/4] Booting primary multi-epoch 2D raw asset generator...
.venv\Scripts\python.exe -m roman_exoplanet_detector.main

echo [3/4] Initializing asynchronous background ML retraining daemon...
start /b "" .venv\Scripts\python.exe -m roman_exoplanet_detector.machine_learning.retrain_worker

echo [4/4] Launching authenticated astronomer review portal web server...
echo ---------------------------------------------------------------------
echo Pipeline boot sequence completed successfully! Press Ctrl+C to stop.
echo ---------------------------------------------------------------------

:: --- EXTENSION CORRECTION STEP ---
:: Move the terminal worker directory path directly down into your source folder 
:: right before spawning the dashboard so that all 'config/serviceAccountKey.json' 
:: lookups evaluate with absolute precision.
cd roman_exoplanet_detector

..\.venv\Scripts\streamlit.exe run src/roman_exoplanet_detector/review/dashboard.py
