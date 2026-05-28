@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║          HeightAI 智能身高识别系统 - 环境安装              ║
echo ╠════════════════════════════════════════════════════════════╣
echo ║  版本: 1.0.0                                               ║
echo ║  支持: Windows 10/11                                       ║
echo ║  Python: 3.8 - 3.11 (推荐 3.10)                            ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:: ============================================================
:: 步骤1: 检查 Python
:: ============================================================
echo [步骤 1/7] 检查 Python 环境...
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8-3.11
    echo.
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    goto :end_error
)

:: 获取Python版本
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [信息] 检测到 Python 版本: %PYVER%

:: 检查Python版本是否在3.8-3.11范围
python -c "import sys; exit(0 if (3,8) <= sys.version_info[:2] <= (3,11) else 1)" 2>nul
if errorlevel 1 (
    echo [警告] Python 版本建议使用 3.8-3.11，当前版本可能有兼容性问题
    echo        继续安装请按任意键，或 Ctrl+C 取消
    pause >nul
)

echo [完成] Python 环境检查通过
echo.

:: ============================================================
:: 步骤2: 创建虚拟环境
:: ============================================================
echo [步骤 2/7] 创建虚拟环境...
echo.

if exist "venv\Scripts\python.exe" (
    echo [信息] 虚拟环境已存在，跳过创建
) else (
    echo [信息] 正在创建虚拟环境 venv...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        echo        请确保已安装 Python 并有足够权限
        goto :end_error
    )
    echo [完成] 虚拟环境创建成功
)
echo.

:: ============================================================
:: 步骤3: 激活虚拟环境
:: ============================================================
echo [步骤 3/7] 激活虚拟环境...
echo.

call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败
    goto :end_error
)
echo [完成] 虚拟环境已激活
echo.

:: ============================================================
:: 步骤4: 升级 pip
:: ============================================================
echo [步骤 4/7] 升级 pip 和基础工具...
echo.

python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [警告] 使用清华源失败，尝试默认源...
    python -m pip install --upgrade pip setuptools wheel
)
echo [完成] pip 升级完成
echo.

:: ============================================================
:: 步骤5: 安装核心依赖
:: ============================================================
echo [步骤 5/7] 安装核心依赖库...
echo.

echo ┌────────────────────────────────────────────────────────────┐
echo │  即将安装以下核心库:                                        │
echo │                                                            │
echo │  • MediaPipe      - AI 人体姿态估计                        │
echo │  • OpenCV         - 图像处理                               │
echo │  • NumPy          - 科学计算                               │
echo │  • Flask          - Web 服务框架                           │
echo │  • Flask-SocketIO - 实时通信                               │
echo │  • Loguru         - 日志系统                               │
echo │  • PyYAML         - 配置管理                               │
echo └────────────────────────────────────────────────────────────┘
echo.

echo [信息] 正在安装核心依赖（使用清华镜像加速）...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

if errorlevel 1 (
    echo [警告] 清华镜像安装失败，尝试默认源...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 核心依赖安装失败
        goto :end_error
    )
)

echo [完成] 核心依赖安装成功
echo.

:: ============================================================
:: 步骤6: 安装可选依赖
:: ============================================================
echo [步骤 6/7] 检查可选依赖...
echo.

:: 检查 RealSense
echo [信息] 检查 Intel RealSense SDK...
pip show pyrealsense2 >nul 2>&1
if errorlevel 1 (
    echo [提示] pyrealsense2 未安装
    echo        如果您有 Intel RealSense 深度相机，可以手动安装:
    echo        pip install pyrealsense2
    echo        没有深度相机也可以使用普通 USB 摄像头运行
) else (
    echo [完成] pyrealsense2 已安装
)
echo.

:: 检查 onnxruntime-gpu
echo [信息] 检查 ONNX Runtime GPU...
pip show onnxruntime-gpu >nul 2>&1
if errorlevel 1 (
    echo [提示] onnxruntime-gpu 未安装
    echo        如果您有 NVIDIA GPU 且已安装 CUDA，可以手动安装:
    echo        pip install onnxruntime-gpu
    echo        否则系统将使用 CPU 版本
) else (
    echo [完成] onnxruntime-gpu 已安装
)
echo.

:: ============================================================
:: 步骤7: 验证安装
:: ============================================================
echo [步骤 7/7] 验证安装结果...
echo.

echo [信息] 检查关键模块...
set CHECK_OK=1

python -c "import cv2; print(f'  OpenCV: {cv2.__version__}')" 2>nul
if errorlevel 1 (
    echo   [错误] OpenCV 导入失败
    set CHECK_OK=0
)

python -c "import mediapipe; print(f'  MediaPipe: OK')" 2>nul
if errorlevel 1 (
    echo   [错误] MediaPipe 导入失败
    set CHECK_OK=0
)

python -c "import numpy; print(f'  NumPy: {numpy.__version__}')" 2>nul
if errorlevel 1 (
    echo   [错误] NumPy 导入失败
    set CHECK_OK=0
)

python -c "import flask; print(f'  Flask: OK')" 2>nul
if errorlevel 1 (
    echo   [错误] Flask 导入失败
    set CHECK_OK=0
)

python -c "import flask_socketio; print(f'  Flask-SocketIO: OK')" 2>nul
if errorlevel 1 (
    echo   [错误] Flask-SocketIO 导入失败
    set CHECK_OK=0
)

python -c "import scipy; print(f'  SciPy: OK')" 2>nul
if errorlevel 1 (
    echo   [错误] SciPy 导入失败
    set CHECK_OK=0
)

echo.

if "%CHECK_OK%"=="1" (
    echo ╔════════════════════════════════════════════════════════════╗
    echo ║                  安装成功完成!                              ║
    echo ╠════════════════════════════════════════════════════════════╣
    echo ║                                                            ║
    echo ║  下一步操作:                                               ║
    echo ║                                                            ║
    echo ║  1. 启动系统:                                              ║
    echo ║     双击运行 "智能数据测试+存储.bat"                        ║
    echo ║     或在命令行运行: python app.py                          ║
    echo ║                                                            ║
    echo ║  2. 打开浏览器访问:                                        ║
    echo ║     http://localhost:5000                                  ║
    echo ║                                                            ║
    echo ║  3. 验证摄像头:                                            ║
    echo ║     双击运行 "验证相机.bat"                                ║
    echo ║                                                            ║
    echo ╚════════════════════════════════════════════════════════════╝
    goto :end_ok
) else (
    echo [错误] 部分模块验证失败，请检查错误信息
    goto :end_error
)

:: ============================================================
:: 结束处理
:: ============================================================
:end_error
echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║                    安装失败                                 ║
echo ╠════════════════════════════════════════════════════════════╣
echo ║  常见问题解决方案:                                          ║
echo ║                                                            ║
echo ║  1. 网络问题: 检查网络连接，或使用 VPN                      ║
echo ║  2. 权限问题: 以管理员身份运行此脚本                        ║
echo ║  3. Python版本: 确保使用 Python 3.8-3.11                   ║
echo ║  4. 虚拟环境: 删除 venv 文件夹后重新运行                    ║
echo ║                                                            ║
echo ║  如需帮助，请查看 README.md 或联系开发者                    ║
echo ╚════════════════════════════════════════════════════════════╝
pause
exit /b 1

:end_ok
echo.
pause
exit /b 0
