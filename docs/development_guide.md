# 开发指南

## 1. 开发规范

### 1.1 代码风格

遵循 **PEP 8** 规范：

```python
class HolisticEstimator:
    """Holistic 姿态估计器类。"""
    
    def __init__(self, model_complexity: int = 1):
        """
        初始化估计器。
        
        Args:
            model_complexity: 模型复杂度，0/1/2
        """
        self.model_complexity = model_complexity
    
    def detect(self, image: np.ndarray, timestamp: float) -> HolisticResult:
        """检测人体姿态和手部。"""
        pass
```

### 1.2 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `CoordinateTransformer`, `StabilityDetector` |
| 函数名 | snake_case | `get_stability()`, `calculate_height()` |
| 变量名 | snake_case | `depth_frame`, `frame_count` |
| 常量 | UPPER_SNAKE | `POSE_CONNECTIONS`, `HAND_CONNECTIONS` |
| 私有成员 | 前缀下划线 | `_start_realsense()` |

### 1.3 类型注解

所有公开函数必须有类型注解：

```python
from typing import List, Dict, Optional, Tuple

def _extract_required_fields(
    measurement_values_cm: Optional[Dict]
) -> Optional[Dict[str, float]]:
    pass
```

---

## 2. 项目结构

### 2.1 核心模块

```
src/
├── core/                      # 核心算法层
│   ├── holistic_estimator.py  # MediaPipe Holistic 估计器
│   ├── coordinate_transformer.py  # 身体坐标变换
│   ├── hand_coordinate_transformer.py  # 手部坐标变换
│   ├── skeleton.py            # 3D骨骼模型
│   ├── stability_detector.py  # 稳定性检测器
│   ├── constants.py           # 常量定义
│   ├── camera_controller.py   # 相机控制器（新增）
│   ├── measurement_collector.py  # 测量采集器（新增）
│   ├── measurement_engine.py  # 测量引擎（新增）
│   ├── depth_processor.py     # 深度处理器（新增）
│   ├── adaptive_depth_sampler.py  # 自适应深度采样（新增）
│   ├── keypoint_stabilizer.py # 关键点稳定器（新增）
│   ├── biomechanical_constraints.py  # 生物力学约束（新增）
│   └── height_calibrator.py   # 身高校准器（新增）
│
├── hardware/                  # 硬件接口层
│   ├── camera_manager.py      # 相机管理器
│   └── frame_set.py           # 帧数据结构
│
├── measurement/               # 测量算法
│   └── linear_measurements.py # 线性测量
│
├── visualization/             # 可视化
│   └── ui_renderer.py         # UI渲染
│
├── output/                    # 输出模块
│   └── data_exporter.py       # 数据导出
│
├── utils/                     # 工具函数
│   └── ...
│
└── main.py                    # 程序入口 ⭐
```

### 2.2 脚本目录

```
scripts/
├── rtmpose_demo.py            # 旧版演示程序（已迁移到 src/main.py）
├── verify_environment.py      # 环境验证
├── verify_camera.py           # 相机验证
├── verify_mediapipe.py        # MediaPipe验证
├── verify_mediapipe.py        # MediaPipe验证
└── ...                        # 其他测试脚本
```

**注意**：主程序入口已从 `scripts/rtmpose_demo.py` 迁移到 `src/main.py`

---

## 3. 开发流程

### 3.1 环境准备

```bash
# 激活虚拟环境
.\venv\Scripts\activate

# 安装开发依赖
pip install -r requirements-dev.txt
```

### 3.2 运行主程序

```bash
# 方式一：运行主程序（推荐）
python src/main.py

# 方式二：使用批处理文件（Windows）
智能数据测试+存储.bat
```

### 3.3 按键操作

| 按键 | 功能 |
|------|------|
| `q` | 退出程序 |
| `r` | 重启摄像头 |
| `c` | 手动触发采集（需已站稳） |
| `h` | 设置身高校准 |

---

## 4. 核心代码说明

### 4.1 状态机实现

```python
# 状态定义
capture_state = "monitoring"  # monitoring | countdown | sampling | done

# 状态转换
if capture_state == "monitoring":
    if stability.is_stable and stable_progress >= 1.0:
        capture_state = "countdown"
        countdown_start_ts = timestamp

elif capture_state == "countdown":
    if not stability.is_stable:
        capture_state = "monitoring"  # 回退
    elif (timestamp - countdown_start_ts) >= countdown_seconds:
        capture_state = "sampling"

elif capture_state == "sampling":
    if len(samples) >= sampling_target_frames:
        # 融合数据并保存
        aggregated = _aggregate_samples(samples)
        capture_state = "done"
```

### 4.2 数据融合算法

```python
def _mean_drop_min_max(values: List[float]) -> float:
    """去掉最大最小值后取平均"""
    if len(values) <= 2:
        return sum(values) / len(values)
    vals = sorted(values)
    trimmed = vals[1:-1]  # 去掉首尾
    return sum(trimmed) / len(trimmed)
```

### 4.3 3D坐标变换

```python
# 身体关键点 2D → 3D
points_3d = transformer.transform_with_filter(
    result.pose,
    depth_image,
    image_size=(image_width, image_height),
    min_visibility=0.5,
    depth_range=(0.3, 4.0),
    use_enhanced=True
)

# 手部关键点融合
if result.left_hand and result.left_hand.detected:
    left_hand_3d = hand_transformer.hand_landmarks_to_3d(
        result.left_hand,
        depth_image,
        (image_width, image_height),
        body_wrist_depth=left_body_wrist_depth
    )
    # 替换身体关键点中的手部位置
    points_3d[19] = left_hand_3d[5]
    points_3d[17] = left_hand_3d[17]
    points_3d[21] = left_hand_3d[1]
```

---

## 5. 调试技巧

### 5.1 查看检测状态

程序界面显示：
- Backend: MediaPipe
- FPS: 实时帧率
- Body/Hand 检测状态
- 稳定性进度条
- 当前状态消息

### 5.2 测量数据面板

右侧面板实时显示 20 个骨骼段测量值（仅 RealSense 模式）。

### 5.3 常见问题

**相机无法初始化**
```python
# 检查 RealSense 连接
import pyrealsense2 as rs
ctx = rs.context()
print(f"检测到 {len(ctx.devices)} 个设备")
```

**MediaPipe 检测不稳定**
- 确保光线充足
- 保持 1.5-2.5 米距离
- 避免背景杂乱

---

## 6. 常用命令

```bash
# 运行主程序（推荐）
python src/main.py

# 或使用批处理文件（Windows）
智能数据测试+存储.bat

# 验证相机
python scripts/verify_camera.py

# 验证 MediaPipe
python scripts/verify_mediapipe.py

# 快速相机测试
python scripts/quick_camera_test.py
```

**注意**：`verify_environment.py` 脚本当前不存在，如需验证环境请直接运行主程序测试。

---

## 7. 数据输出

测量数据保存到 `data/sessions/` 目录：

```
data/sessions/
├── 2025-12-25 10-30-00.json   # 测量数据
├── 2025-12-25 10-30-00.txt    # 可读报告
└── 2025-12-25 10-30-00.jpg    # 截图
```
