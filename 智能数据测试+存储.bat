@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║          HeightAI 智能身高识别系统 - 启动中...             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\python.exe" (
    echo [错误] 虚拟环境不存在，请先运行 "安装环境.bat"
    echo.
    pause
    exit /b 1
)

:: 激活虚拟环境
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败
    pause
    exit /b 1
)

echo [信息] 虚拟环境已激活
echo [信息] 正在启动系统...
echo.

:: 设置环境变量（可选，用于微调测量参数）
if "%MEAS_HEIGHT_SCALE%"=="" set MEAS_HEIGHT_SCALE=1.043
if "%MEAS_LINEAR_SCALE%"=="" set MEAS_LINEAR_SCALE=1.043
if "%MEAS_MIN_VISIBILITY%"=="" set MEAS_MIN_VISIBILITY=0.4

:: 启动主程序
python "src\main.py"

pause
