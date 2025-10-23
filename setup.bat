@echo off
REM ============================================================================
REM  Setup script for the Local Tesseract OCR API
REM
REM  This script will:
REM  1. Create a Python virtual environment.
REM  2. Install the required Python packages.
REM  3. Guide you to install the Tesseract OCR engine.
REM ============================================================================

echo [1/3] Creating Python virtual environment...
if not exist venv (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment. Please ensure Python is installed.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment 'venv' already exists.
)

echo.
echo [2/3] Installing Python dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python packages.
    pause
    exit /b 1
)

echo.
echo [3/3] Checking for Tesseract OCR installation...
python download_models.py
if %errorlevel% neq 0 (
    echo An error occurred during the Tesseract check.
    pause
    exit /b 1
)

echo.
echo =================================================
echo  Setup Complete!
echo =================================================
echo If Tesseract is installed correctly, you can now run the API
echo using the 'run.bat' script.
echo.
pause
