# CycleFitting 优化方案

## 1. 优化目标

| 指标 | 当前状态 | 目标 |
|------|----------|------|
| 测量精度 | ±5-10cm | ±1-2cm |
| 帧率 | 10-15 FPS | ≥25 FPS |
| 系统延迟 | 200-300ms | <100ms |

---

## 2. 详细设计文档

| 阶段 | 文档 | 状态 |
|------|------|------|
| 第一阶段 | 深度数据优化 | ✅ 已完成 |
| 第二阶段 | 关键点稳定 | ✅ 已完成 |
| 第三阶段 | 生物力学约束 | ✅ 已完成 |
| 第四阶段 | 性能优化 | 📝 待实施 |

---

## 3. 优化策略概览

### 3.1 深度数据优化（优先级：高）✅ 已完成

**问题**：2D关键点映射到3D深度时存在系统误差，是测量误差的主要来源。

**实现方案**：

1. **RealSense 滤镜链优化** ✅
   - 实现：空间滤波 + 时间滤波 + 孔洞填充
   - 位置：`src/core/camera_controller.py`
   - 配置：`config/depth_optimization.yaml`

2. **自适应深度采样** ✅
   - 实现：基于身体部位的自适应窗口采样
   - 位置：`src/core/adaptive_depth_sampler.py`

3. **深度图增强** ✅
   - 实现：无效值填充 + 边缘保持滤波
   - 位置：`src/core/depth_processor.py`

### 3.2 关键点稳定（优先级：高）✅ 已完成

**问题**：关键点抖动导致测量值不稳定。

**实现方案**：

1. **One-Euro 滤波器** ✅
   - 实现：对2D和3D关键点应用自适应滤波
   - 位置：`src/core/keypoint_stabilizer.py`
   - 配置：`config/keypoint_stabilization.yaml`

2. ~~卡尔曼滤波器~~ ❌ 未实现
   - 原计划集成到 `keypoint_filter.py`，但当前使用 One-Euro 已满足需求

### 3.3 生物力学约束（优先级：中）✅ 已完成

**问题**：测量值可能超出人体合理范围。

**实现方案**：

1. **人体比例约束** ✅
   - 实现：基于身高的比例约束验证
   - 位置：`src/core/biomechanical_constraints.py`

2. **左右对称性约束** ✅
   - 实现：对称性检查和评分
   - 位置：`src/core/biomechanical_constraints.py`

3. ~~T-Pose 校准~~ ❌ 未实现
   - 原计划的 `calibration_system.py` 未开发
   - 替代方案：使用身高校准器 (`src/core/height_calibrator.py`)

### 3.4 性能优化（优先级：中）📝 待实施

**问题**：处理速度影响实时性。

**计划方案**：

1. **向量化计算**
   - 当前：部分循环处理
   - 优化：使用 NumPy 向量化操作

2. **测量频率控制**
   - 当前：每0.2秒计算一次
   - 优化：可配置的测量间隔

3. **可选的 Numba JIT**
   - 当前：无
   - 优化：对热点函数使用 JIT 编译

---

## 4. 实施计划与任务跟踪

### 第一阶段：深度数据优化 ✅ 已完成

| 任务ID | 任务名称 | 状态 |
|--------|----------|------|
| T1.1 | 创建滤镜链配置类 | ✅ 完成 |
| T1.2 | 修改 CameraController 添加滤镜链 | ✅ 完成 |
| T1.3 | 创建 DepthProcessor 类 | ✅ 完成 |
| T1.4 | 实现深度图增强方法 | ✅ 完成 |
| T1.5 | 创建 AdaptiveDepthSampler 类 | ✅ 完成 |
| T1.6 | 实现身体部位采样策略 | ✅ 完成 |
| T1.7 | 实现时序一致性检查 | ✅ 完成 |
| T1.8 | 集成到 MeasurementEngine | ✅ 完成 |
| T1.9 | 添加配置文件支持 | ✅ 完成 |

### 第二阶段：关键点稳定 ✅ 已完成

| 任务ID | 任务名称 | 状态 |
|--------|----------|------|
| T2.1 | 创建 StabilizerConfig 配置类 | ✅ 完成 |
| T2.2 | 实现 LowPassFilter 类 | ✅ 完成 |
| T2.3 | 实现 OneEuroFilter 类 | ✅ 完成 |
| T2.4 | 创建 KeypointStabilizer 类 | ✅ 完成 |
| T2.5 | 实现 stabilize_2d 方法 | ✅ 完成 |
| T2.6 | 实现 stabilize_3d 方法 | ✅ 完成 |
| T2.7 | 实现置信度自适应调整 | ✅ 完成 |
| T2.8 | 实现异常值检测 | ✅ 完成 |
| T2.9 | 集成到主程序 | ✅ 完成 |

### 第三阶段：生物力学约束 ✅ 已完成

| 任务ID | 任务名称 | 状态 |
|--------|----------|------|
| T3.1 | 创建 ConstraintsConfig 配置类 | ✅ 完成 |
| T3.2 | 创建 BiomechanicalConstraints 类 | ✅ 完成 |
| T3.3 | 实现身高估计方法 | ✅ 完成 |
| T3.4 | 实现比例验证方法 | ✅ 完成 |
| T3.5 | 实现对称性检查方法 | ✅ 完成 |
| T3.6 | 实现置信度评分方法 | ✅ 完成 |
| T3.7 | 实现约束校正方法 | ✅ 完成 |
| T3.8 | 集成到 MeasurementEngine | ✅ 完成 |
| T3.9 | 添加 UI 显示 | ✅ 完成 |

---

## 5. 模块设计（实际实现）

### 5.1 深度采样器 (AdaptiveDepthSampler)

```
位置：src/core/adaptive_depth_sampler.py

功能：
- 基于身体部位的自适应窗口大小
- 前景-背景分离
- 多种统计方法（中位数、截尾均值等）
- 时序一致性检查

接口：
- sample(depth_map, x, y, keypoint_id, confidence, depth_scale) -> float
- batch_sample(depth_map, keypoints, keypoint_ids, confidences, depth_scale) -> List[float]
```

### 5.2 关键点稳定器 (KeypointStabilizer)

```
位置：src/core/keypoint_stabilizer.py

功能：
- One-Euro 滤波（2D 和 3D）
- 根据置信度调整滤波强度
- 历史缓冲区备用策略

接口：
- stabilize_2d(keypoint_id, x, y, confidence, timestamp) -> (x, y)
- stabilize_3d(keypoint_id, x, y, z, confidence, timestamp) -> (x, y, z)
- stabilize_points_3d(points_3d, timestamp) -> List[Point3D]
```

### 5.3 生物力学约束 (BiomechanicalConstraints)

```
位置：src/core/biomechanical_constraints.py

功能：
- 人体比例验证
- 左右对称性约束
- 置信度评分

接口：
- estimate_height(bone_lengths_cm) -> float
- validate_proportions(bone_lengths_cm, height_cm) -> Dict[str, ProportionResult]
- check_symmetry(bone_lengths_cm) -> Dict[str, SymmetryResult]
- validate(bone_lengths_cm, provided_height_cm) -> ConstraintsResult
```

### 5.4 深度处理器 (DepthProcessor)

```
位置：src/core/depth_processor.py

功能：
- 深度图增强（无效值填充、边缘保持滤波）
- 集成自适应采样器

接口：
- process(depth_frame, depth_scale) -> np.ndarray
- sample_keypoint_depth(depth_frame, x, y, keypoint_id, confidence, depth_scale) -> float
```

---

## 6. 配置参数

实际配置文件位置：
- `config/default.yaml` - 主配置
- `config/depth_optimization.yaml` - 深度优化配置
- `config/keypoint_stabilization.yaml` - 关键点稳定配置
- `config/biomechanical_constraints.yaml` - 生物力学约束配置

环境变量覆盖（已实现）：
```bash
MEAS_DEPTH_ENHANCEMENT=true/false      # 启用/禁用深度增强
MEAS_KEYPOINT_STABILIZATION=true/false # 启用/禁用关键点稳定
MEAS_BIOMECHANICAL_CONSTRAINTS=true/false # 启用/禁用生物力学约束
MEAS_MIN_VISIBILITY=0.4               # 最小可见性阈值
MEAS_LINEAR_SCALE=1.0                 # 线性缩放系数
MEAS_HEIGHT_SCALE=1.0                 # 身高缩放系数
```

---

## 7. 注意事项

1. **已实现功能**：所有标记为 ✅ 的功能均已实现并测试
2. **未实现功能**：标记为 ❌ 的功能未开发，请勿依赖
3. **配置兼容性**：`config/default.yaml` 已清理，只包含实际可用的配置项
4. **性能监控**：当前帧率约 25-30 FPS（RealSense 模式）

---

## 8. 参考资源

- One-Euro Filter 论文：https://hal.inria.fr/hal-00670496/document
- RealSense 滤镜文档：https://dev.intelrealsense.com/docs/post-processing-filters
- MediaPipe Pose：https://google.github.io/mediapipe/solutions/pose.html
