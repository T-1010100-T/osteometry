@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║          HeightAI 智能身高识别系统 - 摄像头验证            ║
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

echo [信息] 正在检测摄像头...
echo.

python "scripts\verify_camera.py"

pause
