# RealSense D455 人体骨骼识别与测量系统

基于 Intel RealSense D455 深度相机和 MediaPipe Holistic 的实时人体姿态捕捉与3D身体尺寸测量系统。

## 核心功能

### 1. 人体姿态检测
- 使用 MediaPipe Holistic 检测
- 33个身体关键点 + 21×2个手部关键点
- 实时骨骼可视化

### 2. 3D坐标转换
- RealSense 深度相机集成
- 2D关键点 → 3D世界坐标
- 深度数据滤波和校正

### 3. 身体测量
- 身高测量
- 20个骨骼段长度（脊柱、肩部、手臂、腿部各段）
- 多帧采样融合（去最大/最小后取平均）

### 4. 智能数据采集
- 稳定性检测（30帧稳定触发）
- 倒计时采集（2秒）
- 多帧采样（10帧/3秒窗口）
- 自动保存（JSON + TXT + JPG）

## 快速开始

### 环境要求
- Python 3.8 - 3.12（推荐 3.10+）
- Intel RealSense D455 相机（推荐）或普通 USB 摄像头
- Windows 10/11
- USB 3.0 端口

### 安装步骤

**方式一：使用批处理文件（推荐）**
1. 双击 `安装环境.bat` 安装依赖
2. 双击 `智能图像识别.bat` 启动程序

**方式二：命令行**
```bash
# 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动程序
python scripts/rtmpose_demo.py
```

## 使用说明

### 按键操作
- `q` - 退出
- `r` - 重启摄像头
- `c` - 手动触发采集（需已站稳）

### 采集流程
1. 启动程序
2. 站在摄像头前 1.5-2.5 米处
3. 保持稳定，等待稳定性进度条满
4. 倒计时 2 秒后自动采样
5. 采集完成后数据自动保存到 `data/sessions/`

## 项目结构

```
CycleFitting/
├── src/                    # 源代码
│   ├── core/              # 核心算法（姿态估计、坐标变换）
│   ├── hardware/          # 硬件接口
│   └── measurement/       # 测量引擎
├── scripts/               # 脚本
│   └── rtmpose_demo.py    # 主程序 ⭐
├── config/                # 配置文件
├── data/sessions/         # 测量数据输出
└── docs/                  # 文档
```

## 数据输出

测量数据保存到 `data/sessions/` 目录：

| 文件 | 内容 |
|------|------|
| `*.json` | 结构化测量数据（20个骨骼段长度） |
| `*.txt` | 人类可读的测量报告 |
| `*.jpg` | 采集时的截图 |

### 测量项目（20项）

- 身高
- 脊柱：底部到中部、中部到肩部、肩部到头部
- 左臂：肩部到左肩、左肩到左肘、左肘到左腕、左腕到左手
- 右臂：肩部到右肩、右肩到右肘、右肘到右腕、右腕到右手
- 左腿：底部到左髋、左髋到左膝、左膝到左踝、左踝到左脚
- 右腿：底部到右髋、右髋到右膝、右膝到右踝、右踝到右脚

## 技术栈

- **相机SDK**: Intel RealSense SDK 2.0 (pyrealsense2)
- **姿态估计**: MediaPipe Holistic
- **图像处理**: OpenCV
- **科学计算**: NumPy

## 文档

- [架构设计](docs/architecture.md)
- [环境配置](docs/environment_setup.md)
- [项目结构](docs/project_structure.md)
- [开发指南](docs/development_guide.md)
- [API设计](docs/api_design.md)
- [数据流设计](docs/data_flow.md)

## 许可证

MIT License
