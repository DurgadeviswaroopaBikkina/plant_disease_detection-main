@echo off
echo ====================================
echo Plant Disease Detection App
echo ====================================

REM Go to project directory
cd /d "%~dp0"

REM Activate virtual environment
call venv\Scripts\activate

REM Optional: force CPU mode (avoid CUDA warnings)
set CUDA_VISIBLE_DEVICES=-1

REM Run the Flask app
python app.py

pause
