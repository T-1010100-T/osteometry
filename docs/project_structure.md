# 项目目录结构

## 目录树

```
CycleFitting/
│
├── README.md                       # 项目说明
├── requirements.txt                # Python 依赖
├── requirements-dev.txt            # 开发依赖
├── .gitignore                      # Git 忽略规则
│
├── 安装环境.bat                    # 环境安装脚本
├── 智能数据测试+存储.bat           # 主程序启动脚本 ⭐
├── 验证相机.bat                    # 相机验证脚本
│
├── config/                         # 配置文件目录
│   ├── default.yaml               # 默认配置
│   ├── holistic_settings.yaml     # Holistic检测配置
│   ├── smart_collector.yaml       # 智能采集器配置
│   ├── depth_optimization.yaml    # 深度优化配置 ⭐ (优化)
│   ├── keypoint_stabilization.yaml  # 关键点稳定配置 ⭐ (优化)
│   └── biomechanical_constraints.yaml  # 生物力学约束配置 ⭐ (优化)
│
├── src/                            # 源代码目录
│   ├── __init__.py
│   ├── main.py                    # 程序入口 ⭐
│   │
│   ├── core/                      # 核心算法层 ⭐
│   │   ├── holistic_estimator.py  # MediaPipe Holistic 估计器
│   │   ├── coordinate_transformer.py  # 身体坐标变换
│   │   ├── hand_coordinate_transformer.py  # 手部坐标变换
│   │   ├── skeleton.py            # 3D骨骼模型
│   │   ├── stability_detector.py  # 稳定性检测器
│   │   ├── constants.py           # 常量定义
│   │   ├── camera_controller.py   # 相机控制器 ⭐ (新)
│   │   ├── measurement_collector.py  # 测量采集器 ⭐ (新)
│   │   ├── measurement_engine.py  # 测量引擎 ⭐ (新)
│   │   ├── data_aggregator.py     # 数据聚合 ⭐ (新)
│   │   ├── depth_config.py        # 深度优化配置 ⭐ (优化)
│   │   ├── depth_processor.py     # 深度处理器 ⭐ (优化)
│   │   ├── adaptive_depth_sampler.py  # 自适应深度采样 ⭐ (优化)
│   │   ├── stabilizer_config.py   # 关键点稳定配置 ⭐ (优化)
│   │   ├── keypoint_stabilizer.py # 关键点稳定器 ⭐ (优化)
│   │   ├── biomechanical_constraints.py  # 生物力学约束 ⭐ (优化)
│   │   ├── pose_estimator.py      # 姿态估计基类
│   │   ├── keypoint_filter.py     # 关键点滤波
│   │   ├── hand_skeleton.py       # 手部骨骼模型
│   │   ├── hand_result.py         # 手部检测结果
│   │   ├── hand_filter.py         # 手部滤波器
│   │   ├── hand_exceptions.py     # 手部异常定义
│   │   ├── smart_collector.py     # 智能采集器
│   │   ├── smart_collector_types.py  # 采集器类型
│   │   ├── quality_scorer.py      # 质量评分
│   │   ├── frame_fusion.py        # 帧融合
│   │   ├── session_manager.py     # 会话管理
│   │   ├── gesture_recognizer.py  # 手势识别
│   │   └── roi_optimizer.py       # ROI优化
│   │
│   ├── hardware/                  # 硬件接口层
│   │   ├── camera_manager.py      # 相机管理器
│   │   └── frame_set.py           # 帧数据结构
│   │
│   ├── measurement/               # 测量算法
│   │   ├── linear_measurements.py # 线性测量 ⭐
│   │   ├── measurement_engine.py  # 测量引擎
│   │   ├── hand_measurement.py    # 手部测量
│   │   ├── joint_angles.py        # 关节角度
│   │   ├── circumference.py       # 围度测量
│   │   └── calibration.py         # 校准
│   │
│   ├── visualization/             # 可视化模块
│   │   ├── skeleton_drawer.py     # 骨骼绘制
│   │   ├── measurement_overlay.py # 测量叠加
│   │   ├── ui_renderer.py         # UI渲染 ⭐ (新)
│   │   └── viewer_3d.py           # 3D查看器
│   │
│   ├── output/                    # 输出模块
│   │   ├── data_exporter.py       # 数据导出
│   │   └── report_generator.py    # 报告生成
│   │
│   ├── processing/                # 数据处理（空）
│   │
│   └── utils/                     # 工具函数
│       ├── config.py              # 配置加载
│       ├── logger.py              # 日志
│       ├── math_utils.py          # 数学工具
│       └── validators.py          # 验证器
│
├── scripts/                        # 脚本目录
│   ├── rtmpose_demo.py            # 旧版演示程序（已解耦到 src/）
│   ├── bone_length_test.py        # 骨骼长度测试
│   ├── bone_segment_collector.py  # 骨骼段采集
│   ├── convert_json_to_csv.py     # JSON转CSV
│   ├── check_imu.py               # IMU检测
│   ├── check_level.py             # 水平检测
│   ├── verify_environment.py      # 环境验证
│   ├── verify_camera.py           # 相机验证
│   ├── verify_mediapipe.py        # MediaPipe验证
│   ├── patch_mediapipe.py         # MediaPipe修补
│   ├── quick_camera_test.py       # 快速相机测试
│   └── test_camera_only.py        # 纯相机测试
│
├── tests/                          # 测试代码
│
├── data/                           # 数据目录
│   └── sessions/                  # 测量会话数据
│
└── docs/                           # 文档目录
    ├── architecture.md            # 架构设计
    ├── project_structure.md       # 项目结构
    ├── environment_setup.md       # 环境配置
    ├── development_guide.md       # 开发指南
    ├── api_design.md              # API设计
    ├── data_flow.md               # 数据流设计
    ├── testing_plan.md            # 测试方案
    ├── optimization_plan.md       # 优化方案 ⭐ (优化)
    └── optimization/              # 优化详细设计 ⭐ (优化)
        ├── phase1_depth_optimization.md      # 深度数据优化
        ├── phase2_keypoint_stabilization.md  # 关键点稳定
        └── phase3_biomechanical_constraints.md  # 生物力学约束
```

## 核心模块说明

### 新增模块（解耦自 rtmpose_demo.py）

| 模块 | 用途 |
|------|------|
| `src.core.camera_controller` | RealSense/OpenCV 相机管理 |
| `src.core.measurement_collector` | 采集状态机和采样逻辑 |
| `src.core.measurement_engine` | 测量计算引擎 |
| `src.core.data_aggregator` | 数据验证和融合 |
| `src.visualization.ui_renderer` | UI 渲染（面板、进度条、水平仪）|

### 优化模块（提升测量精度）

| 模块 | 用途 |
|------|------|
| `src.core.depth_processor` | 深度图增强和后处理 |
| `src.core.adaptive_depth_sampler` | 自适应深度采样（基于身体部位）|
| `src.core.keypoint_stabilizer` | One-Euro 滤波器稳定关键点 |
| `src.core.biomechanical_constraints` | 人体比例验证和对称性检查 |

### 主程序依赖模块

| 模块 | 用途 |
|------|------|
| `src.core.holistic_estimator` | MediaPipe Holistic 检测 |
| `src.core.camera_controller` | 相机控制 |
| `src.core.measurement_collector` | 采集状态机 |
| `src.core.measurement_engine` | 测量计算 |
| `src.core.stability_detector` | 稳定性检测 |
| `src.visualization.ui_renderer` | UI 渲染 |

## 数据输出

测量数据保存到 `data/sessions/` 目录：

| 文件类型 | 内容 |
|----------|------|
| `*.json` | 结构化测量数据 |
| `*.txt` | 人类可读的测量报告 |
| `*.jpg` | 采集时的截图 |

## 启动方式

```bash
# 命令行（推荐）
python src/main.py

# 或双击批处理文件
智能数据测试+存储.bat
```
