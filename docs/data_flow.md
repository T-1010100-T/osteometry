# 数据流设计文档

## 1. 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              系统数据流                                      │
└─────────────────────────────────────────────────────────────────────────────┘

     RealSense D455 / OpenCV 摄像头
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 原始数据采集                                                 │
│ RGB (640×480×3) + Depth (640×480) [仅RealSense]             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ MediaPipe Holistic 检测                                      │
│ → 33个身体关键点 + 21×2个手部关键点                          │
└────────────────────────────┬────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ 身体3D变换      │ │ 左手3D变换      │ │ 右手3D变换      │
│ (33点)          │ │ (21点)          │ │ (21点)          │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 手部数据融合到身体骨骼                                       │
│ points_3d[19,17,21] ← left_hand_3d[5,17,1]                  │
│ points_3d[20,18,22] ← right_hand_3d[5,17,1]                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Skeleton3D 骨骼模型                                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ LinearMeasurements 测量计算                                  │
│ → 身高 + 20个骨骼段长度                                      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 状态机控制                                                   │
│ monitoring → countdown → sampling → done                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 多帧融合 (去最大/最小后取平均)                               │
│ 10帧采样，3秒窗口                                            │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ 数据保存                                                     │
│ data/sessions/YYYY-MM-DD HH-MM-SS.{json,txt,jpg}            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 各阶段数据格式

### 2.1 原始数据采集

**RealSense 模式**
```python
color_image: np.ndarray  # shape: (480, 640, 3), dtype: uint8, BGR格式
depth_image: np.ndarray  # shape: (480, 640), dtype: uint16, 单位: mm
```

**OpenCV 模式**
```python
color_image: np.ndarray  # shape: (480, 640, 3), dtype: uint8, BGR格式
depth_image = None       # 无深度数据
```

### 2.2 Holistic 检测结果

```python
HolisticResult:
    pose: PoseResult
        landmarks: List[Landmark]   # 33个关键点
        detected: bool
    left_hand: HandResult
        landmarks: List[Landmark]   # 21个关键点
        detected: bool
    right_hand: HandResult
        landmarks: List[Landmark]   # 21个关键点
        detected: bool

Landmark:
    x: float    # 归一化X坐标 [0, 1]
    y: float    # 归一化Y坐标 [0, 1]
    z: float    # 相对深度
    visibility: float  # 可见性 [0, 1]
```

### 2.3 3D坐标变换

**变换公式**
```
Z = depth[v, u] * depth_scale  # 深度值（米）
X = (u - ppx) * Z / fx
Y = (v - ppy) * Z / fy
```

**输出**
```python
points_3d: List[Point3D]  # 33个3D点

Point3D:
    x: float  # 米，相机坐标系X（右为正）
    y: float  # 米，相机坐标系Y（下为正）
    z: float  # 米，相机坐标系Z（前为正，深度）
```

### 2.4 测量结果

```python
measurement_values_cm: Dict[str, float]
# {
#     '身高': 175.2,
#     '脊柱底部到中部': 15.3,
#     '脊柱中部到肩部': 20.1,
#     '脊柱肩部到头部': 18.5,
#     '脊柱肩部到左肩': 12.0,
#     '左肩到左肘': 28.5,
#     '左肘到左腕': 25.2,
#     '左腕到左手': 8.5,
#     ... (共20项)
# }
```

---

## 3. 采样与融合

### 3.1 采样参数

| 参数 | 值 | 说明 |
|------|-----|------|
| required_stable_frames | 30 | 触发采集所需稳定帧数 |
| countdown_seconds | 2.0 | 倒计时时长 |
| sampling_window_seconds | 3.0 | 采样窗口时长 |
| sampling_target_frames | 10 | 目标采样帧数 |
| sampling_interval_seconds | 0.3 | 采样间隔 |

### 3.2 融合算法

```python
def _mean_drop_min_max(values: List[float]) -> float:
    """去掉最大最小值后取平均"""
    if len(values) <= 2:
        return sum(values) / len(values)
    vals = sorted(values)
    trimmed = vals[1:-1]  # 去掉首尾
    return sum(trimmed) / len(trimmed)

def _aggregate_samples(samples: List[Dict[str, float]]) -> Dict[str, float]:
    """聚合多帧采样数据"""
    aggregated = {}
    for key in REQUIRED_MEASUREMENT_FIELDS_CN:
        vals = [s.get(key, 0.0) for s in samples]
        aggregated[key] = _mean_drop_min_max(vals)
    return aggregated
```

---

## 4. 输出文件格式

### 4.1 JSON 格式

```json
{
  "_meta": {
    "timestamp": "2025-12-25T10:30:00.123456",
    "camera_mode": "realsense",
    "stable_frames_required": 30,
    "countdown_seconds": 2.0,
    "sampling_window_seconds": 3.0,
    "sampling_target_frames": 10,
    "sampling_frames_collected": 10,
    "method": "drop_min_max_then_mean"
  },
  "身高": 175.2,
  "脊柱底部到中部": 15.3,
  "脊柱中部到肩部": 20.1,
  "脊柱肩部到头部": 18.5,
  "脊柱肩部到左肩": 12.0,
  "左肩到左肘": 28.5,
  "左肘到左腕": 25.2,
  "左腕到左手": 8.5,
  "脊柱肩部到右肩": 12.1,
  "右肩到右肘": 28.3,
  "右肘到右腕": 25.0,
  "右腕到右手": 8.4,
  "脊柱底部到左髋": 10.2,
  "左髋到左膝": 42.5,
  "左膝到左踝": 38.2,
  "左踝到左脚": 5.5,
  "脊柱底部到右髋": 10.3,
  "右髋到右膝": 42.3,
  "右膝到右踝": 38.0,
  "右踝到右脚": 5.4
}
```

### 4.2 TXT 格式

```
==================================================
        身体测量数据（单位：cm）
==================================================

测量时间	2025-12-25T10:30:00.123456
相机模式	realsense
站稳帧阈值	30
倒计时	2.0s
采样窗口	3.0s
目标采样帧数	10
实际采样帧数	10
融合规则	去最大/最小后取平均

--------------------------------------------------
身高	175.2cm
脊柱底部到中部	15.3cm
脊柱中部到肩部	20.1cm
脊柱肩部到头部	18.5cm
脊柱肩部到左肩	12.0cm
左肩到左肘	28.5cm
左肘到左腕	25.2cm
左腕到左手	8.5cm
脊柱肩部到右肩	12.1cm
右肩到右肘	28.3cm
右肘到右腕	25.0cm
右腕到右手	8.4cm
脊柱底部到左髋	10.2cm
左髋到左膝	42.5cm
左膝到左踝	38.2cm
左踝到左脚	5.5cm
脊柱底部到右髋	10.3cm
右髋到右膝	42.3cm
右膝到右踝	38.0cm
右踝到右脚	5.4cm

==================================================
```

---

## 5. 实时处理流程

### 5.1 单帧处理时间

| 阶段 | 时间 |
|------|------|
| 帧获取 | ~5ms |
| MediaPipe 检测 | ~30ms |
| 3D变换 | ~2ms |
| 测量计算 | ~3ms |
| 可视化绘制 | ~5ms |
| **总计** | ~45ms (~22 FPS) |

### 5.2 深度数据处理

```python
# 深度获取策略
depth_value = transformer.apply_median_filter(depth_image, u, v)

# 有效深度范围
depth_range = (0.3, 4.0)  # 0.3米 ~ 4.0米
```

---

## 6. 错误处理

### 6.1 相机断开

```python
# RealSense 超时处理
try:
    frames = pipeline.wait_for_frames(timeout_ms=1000)
except RuntimeError as e:
    if "Frame didn't arrive" in str(e):
        realsense_timeouts += 1
        if realsense_timeouts >= 5:
            # 重启相机
            pipeline, config, align, transformer, hand_transformer, err = \
                _restart_realsense(pipeline, config)
```

### 6.2 稳定性丢失

当采集过程中稳定性丢失时，状态机回退到 monitoring 状态：

```python
if not stability.is_stable:
    capture_state = "monitoring"
    stability_detector.reset()
    samples = []
```
