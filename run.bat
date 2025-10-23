@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Starting the OCR server...
echo This service will run on http://127.0.0.1:5004
echo Leave this window open. Press Ctrl+C to stop the server.
echo.
python app.py
