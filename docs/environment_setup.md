# 开发环境配置指南

## 1. 系统要求

### 1.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| 内存 | 8GB | 16GB+ |
| 存储 | 10GB 可用空间 | 50GB+ SSD |
| USB | USB 3.0 端口 | USB 3.1+ |

### 1.2 软件要求

| 软件 | 版本 |
|------|------|
| 操作系统 | Windows 10/11 64-bit |
| Python | 3.8 - 3.12（推荐 3.10+） |

### 1.3 深度相机

- **型号**：Intel RealSense D455（推荐）
- **备选**：普通 USB 摄像头（无深度测量功能）

---

## 2. 安装步骤

### 2.1 安装 Intel RealSense SDK（可选）

如果使用 RealSense 深度相机：

1. 下载 SDK：https://github.com/IntelRealSense/librealsense/releases
2. 运行 `Intel.RealSense.SDK-WIN10-*.exe`
3. 验证：打开 RealSense Viewer 检查相机连接

### 2.2 配置 Python 环境

**方式一：使用批处理文件（推荐）**

双击 `安装环境.bat`

**方式二：命令行**

```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\activate

# 升级 pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 2.3 MediaPipe 中文路径问题修复

如果项目路径包含中文字符，需要创建 Junction 链接：

以管理员身份运行 PowerShell：

```powershell
New-Item -ItemType Junction -Path "C:\mediapipe_temp" -Target "你的项目路径\venv\Lib\site-packages"

[Environment]::SetEnvironmentVariable("MEDIAPIPE_JUNCTION_PATH", "C:\mediapipe_temp", "User")
```

---

## 3. 验证安装

```powershell
# 验证相机
python scripts/verify_camera.py

# 验证 MediaPipe
python scripts/verify_mediapipe.py
```

---

## 4. 运行程序

```powershell
.\venv\Scripts\activate
python src/main.py
```

按键操作：
- `q` - 退出
- `r` - 重启摄像头
- `c` - 手动触发采集

---

## 5. 常见问题

### 5.1 相机无法检测

1. 确保使用 USB 3.0 端口
2. 尝试不同的 USB 线缆
3. 更新相机固件（通过 RealSense Viewer）

### 5.2 MediaPipe 初始化失败

1. 检查路径是否包含中文
2. 创建 Junction 链接（见 2.3）
3. 重新安装：`pip uninstall mediapipe && pip install mediapipe`

---

## 6. 主要依赖

```
pyrealsense2>=2.50.0    # Intel RealSense SDK
mediapipe>=0.10.0       # MediaPipe 姿态估计
opencv-contrib-python>=4.8.0  # 图像处理
numpy>=1.24.0           # 科学计算
pillow>=10.0.0          # 图像处理
pyyaml>=6.0             # 配置管理
```
