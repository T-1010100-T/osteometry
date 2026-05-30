@echo off
cd /d "%~dp0"

echo.
echo  HeightAI - Height Measurement System
echo  http://localhost:5000
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run install.bat first.
    pause
    exit /b 1
)

cmd /k "venv\Scripts\activate.bat && python app.py"
