# 系统架构设计文档

## 1. 架构概览

### 1.1 分层架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层 (Application)                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              MediaPipe 演示 (rtmpose_demo.py)        │    │
│  │  状态机: monitoring → countdown → sampling → done    │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    业务逻辑层 (Business)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 稳定性检测 (StabilityDetector)                       │    │
│  │ 多帧采样融合 (去最大/最小后取平均)                    │    │
│  │ 会话数据保存 (JSON/TXT/JPG)                          │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    测量算法层 (Measurement)                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │    身体测量引擎 (LinearMeasurements)                  │    │
│  │    - 身高、20个骨骼段长度                             │    │
│  │    - 脊柱、肩部、手臂、腿部各段                       │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    核心算法层 (Core)                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │ Holistic    │ │ 3D坐标变换  │ │ 骨骼模型    │            │
│  │ Estimator   │ │ Coordinate  │ │ Skeleton3D  │            │
│  │ (身体+双手) │ │ Transformer │ │             │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│  ┌─────────────┐ ┌─────────────┐                            │
│  │ 手部坐标    │ │ 稳定性检测  │                            │
│  │ Hand        │ │ Stability   │                            │
│  │ Transformer │ │ Detector    │                            │
│  └─────────────┘ └─────────────┘                            │
├─────────────────────────────────────────────────────────────┤
│                    硬件接口层 (Hardware)                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │ RealSense   │ │  OpenCV     │ │ 帧数据结构  │            │
│  │   D455      │ │  摄像头     │ │ FrameSet    │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心设计原则

| 原则 | 描述 |
|------|------|
| **状态机驱动** | 通过状态机控制采集流程 |
| **双相机支持** | 支持 RealSense 深度相机和普通 OpenCV 摄像头 |
| **模块化** | 各层独立，通过接口通信 |
| **实时处理** | 30 FPS 实时姿态检测和测量 |

---

## 2. 数据采集状态机

### 2.1 状态机设计

```
┌──────────────┐
│  monitoring  │ ← 监测稳定性，等待站稳
└────┬─────────┘
     │ 稳定性达标 (stable_frames >= 30)
     ▼
┌──────────────┐
│  countdown   │ ← 倒计时 2 秒
└────┬─────────┘
     │ 倒计时结束
     ▼
┌──────────────┐
│  sampling    │ ← 采集 10 帧数据（3秒窗口）
└────┬─────────┘
     │ 采集完成
     ▼
┌──────────────┐
│    done      │ ← 保存数据，显示冻结画面
└──────────────┘
     │ 按 'r' 重启
     ▼
   返回 monitoring
```

### 2.2 状态转换条件

| 当前状态 | 目标状态 | 触发条件 |
|----------|----------|----------|
| monitoring | countdown | 稳定帧数 ≥ 30 且检测到人体且有测量数据 |
| countdown | sampling | 倒计时 2 秒结束 |
| countdown | monitoring | 稳定性丢失 |
| sampling | done | 采集 10 帧或 3 秒窗口结束 |
| sampling | monitoring | 稳定性丢失 |
| done | monitoring | 按 'r' 重启 |

---

## 3. 姿态检测

### 3.1 MediaPipe Holistic

使用 MediaPipe Holistic 同时检测身体和双手：

```python
HolisticEstimator
├── detect(image, timestamp) -> HolisticResult
│   ├── pose: PoseResult        # 33个身体关键点
│   ├── left_hand: HandResult   # 21个左手关键点
│   └── right_hand: HandResult  # 21个右手关键点
```

### 3.2 关键点定义

**MediaPipe 身体关键点（33点）**

演示程序忽略的关键点（面部细节）：1-10, 17-22

主要使用的关键点：
```
0: nose
11: left_shoulder     12: right_shoulder
13: left_elbow        14: right_elbow
15: left_wrist        16: right_wrist
23: left_hip          24: right_hip
25: left_knee         26: right_knee
27: left_ankle        28: right_ankle
```

**手部关键点（21点）**
```
0: wrist
1-4: thumb (CMC, MCP, IP, TIP)
5-8: index finger
9-12: middle finger
13-16: ring finger
17-20: pinky finger
```

---

## 4. 测量算法

### 4.1 骨骼段测量

系统测量 20 个骨骼段长度（单位：cm）：

| 测量项 | 说明 |
|--------|------|
| 身高 | 头顶到脚底垂直距离 |
| 脊柱底部到中部 | spine_base_to_spine_mid |
| 脊柱中部到肩部 | spine_mid_to_spine_shoulder |
| 脊柱肩部到头部 | spine_shoulder_to_head |
| 脊柱肩部到左肩 | spine_shoulder_to_shoulder_left |
| 左肩到左肘 | shoulder_left_to_elbow_left |
| 左肘到左腕 | elbow_left_to_wrist_left |
| 左腕到左手 | wrist_left_to_hand_left |
| 脊柱肩部到右肩 | spine_shoulder_to_shoulder_right |
| 右肩到右肘 | shoulder_right_to_elbow_right |
| 右肘到右腕 | elbow_right_to_wrist_right |
| 右腕到右手 | wrist_right_to_hand_right |
| 脊柱底部到左髋 | spine_base_to_hip_left |
| 左髋到左膝 | hip_left_to_knee_left |
| 左膝到左踝 | knee_left_to_ankle_left |
| 左踝到左脚 | ankle_left_to_foot_left |
| 脊柱底部到右髋 | spine_base_to_hip_right |
| 右髋到右膝 | hip_right_to_knee_right |
| 右膝到右踝 | knee_right_to_ankle_right |
| 右踝到右脚 | ankle_right_to_foot_right |

### 4.2 手部数据融合

当检测到手部时，使用手部关键点替换身体关键点中的手部位置：
- 左手：points_3d[19], points_3d[17], points_3d[21]
- 右手：points_3d[20], points_3d[18], points_3d[22]

---

## 5. 数据流

```
RealSense/OpenCV Camera
      │
      ▼
┌─────────────┐
│ RGB + Depth │  (RealSense: 640x480, 30fps)
└─────────────┘
      │
      ▼
┌─────────────────────┐
│ HolisticEstimator   │
│ (MediaPipe Holistic)│
└─────────────────────┘
      │
      ├──> PoseResult (33 landmarks)
      ├──> LeftHandResult (21 landmarks)
      └──> RightHandResult (21 landmarks)
      │
      ▼
┌─────────────────────┐
│ CoordinateTransformer│  (仅 RealSense 模式)
│ (2D → 3D)           │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Skeleton3D          │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ LinearMeasurements  │
│ (骨骼段长度计算)     │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 稳定性检测 + 采样    │
│ (状态机控制)         │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ 数据保存            │
│ data/sessions/      │
│ (JSON + TXT + JPG)  │
└─────────────────────┘
```

---

## 6. 输出格式

### 6.1 JSON 数据

```json
{
  "_meta": {
    "timestamp": "2025-12-25T10:30:00",
    "camera_mode": "realsense",
    "stable_frames_required": 30,
    "countdown_seconds": 2,
    "sampling_window_seconds": 3,
    "sampling_target_frames": 10,
    "sampling_frames_collected": 10,
    "method": "drop_min_max_then_mean"
  },
  "身高": 175.2,
  "脊柱底部到中部": 15.3,
  "脊柱中部到肩部": 20.1,
  ...
}
```

### 6.2 TXT 报告

```
==================================================
        身体测量数据（单位：cm）
==================================================

测量时间	2025-12-25T10:30:00
相机模式	realsense
站稳帧阈值	30
倒计时	2s
采样窗口	3s
目标采样帧数	10
实际采样帧数	10
融合规则	去最大/最小后取平均

--------------------------------------------------
身高	175.2cm
脊柱底部到中部	15.3cm
...
==================================================
```

---

## 7. 性能指标

| 指标 | 数值 |
|------|------|
| 帧率 | ~30 FPS |
| 检测延迟 | ~50ms |
| 测量计算 | <10ms |
| 内存使用 | ~500MB |

**硬件要求：**
- RealSense D455 深度相机（推荐）或普通 USB 摄像头
- USB 3.0 端口
- CPU: Intel i5 或更高
