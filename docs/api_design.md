# API 设计文档

## 1. 核心算法层 API

### 1.1 HolisticEstimator

```python
class HolisticEstimator:
    """MediaPipe Holistic 姿态估计器"""
    
    def __init__(self):
        """初始化估计器"""
    
    def detect(self, image: np.ndarray, timestamp: float = None) -> HolisticResult:
        """
        检测图像中的人体姿态和手部
        
        Args:
            image: RGB图像，shape (H, W, 3)
            timestamp: 时间戳
        
        Returns:
            HolisticResult: 检测结果
        """
    
    def close(self) -> None:
        """释放资源"""
```

### 1.2 HolisticResult

```python
@dataclass
class HolisticResult:
    pose: PoseResult           # 33个身体关键点
    left_hand: HandResult      # 21个左手关键点
    right_hand: HandResult     # 21个右手关键点

@dataclass
class PoseResult:
    landmarks: List[Landmark]  # 关键点列表
    detected: bool             # 是否检测到

@dataclass
class HandResult:
    landmarks: List[Landmark]  # 关键点列表
    detected: bool             # 是否检测到

@dataclass
class Landmark:
    x: float          # 归一化X坐标 (0-1)
    y: float          # 归一化Y坐标 (0-1)
    z: float          # 相对深度
    visibility: float # 可见性 (0-1)
```

---

## 2. 坐标变换 API

### 2.1 CoordinateTransformer

```python
class CoordinateTransformer:
    """身体坐标变换器"""
    
    def __init__(self, intrinsics: Intrinsics, depth_scale: float = 0.001):
        """
        初始化变换器
        
        Args:
            intrinsics: 相机内参
            depth_scale: 深度缩放因子
        """
    
    def transform_with_filter(
        self,
        pose: PoseResult,
        depth_image: np.ndarray,
        image_size: Tuple[int, int],
        min_visibility: float = 0.5,
        depth_range: Tuple[float, float] = (0.3, 4.0),
        use_enhanced: bool = True
    ) -> List[Point3D]:
        """
        将2D关键点转换为3D坐标（带滤波）
        
        Args:
            pose: 姿态检测结果
            depth_image: 深度图
            image_size: 图像尺寸 (width, height)
            min_visibility: 最小可见性阈值
            depth_range: 有效深度范围 (min, max) 米
            use_enhanced: 是否使用增强滤波
        
        Returns:
            List[Point3D]: 33个3D坐标点
        """
    
    def apply_median_filter(
        self, 
        depth_image: np.ndarray, 
        u: int, 
        v: int
    ) -> Optional[float]:
        """对深度值应用中值滤波"""
```

### 2.2 HandCoordinateTransformer

```python
class HandCoordinateTransformer:
    """手部坐标变换器"""
    
    def __init__(self, intrinsics: Intrinsics, depth_scale: float = 0.001):
        """初始化变换器"""
    
    def hand_landmarks_to_3d(
        self,
        hand: HandResult,
        depth_image: np.ndarray,
        image_size: Tuple[int, int],
        body_wrist_depth: Optional[float] = None
    ) -> List[Point3D]:
        """
        将手部2D关键点转换为3D坐标
        
        Args:
            hand: 手部检测结果
            depth_image: 深度图
            image_size: 图像尺寸
            body_wrist_depth: 身体手腕深度（用于参考）
        
        Returns:
            List[Point3D]: 21个3D坐标点
        """
```

---

## 3. 骨骼模型 API

### 3.1 Skeleton3D

```python
class Skeleton3D:
    """3D骨骼模型"""
    
    def __init__(self, joints: Dict[str, Point3D]):
        """初始化骨骼模型"""
    
    @classmethod
    def from_points(cls, points: List[Point3D]) -> 'Skeleton3D':
        """从点列表创建骨骼模型"""
    
    def get_joint(self, name: str) -> Point3D:
        """获取指定关节的3D坐标"""

@dataclass
class Point3D:
    x: float  # 单位: 米
    y: float
    z: float
    confidence: float = 1.0
```

---

## 4. 测量算法 API

### 4.1 LinearMeasurements

```python
class LinearMeasurements:
    """线性尺寸测量"""
    
    @staticmethod
    def calculate_height(skeleton: Skeleton3D) -> float:
        """
        计算身高
        
        Returns:
            身高（米）
        """
    
    @staticmethod
    def calculate_bike_fitting_bones(skeleton: Skeleton3D) -> Dict[str, float]:
        """
        计算自行车拟合所需的骨骼段长度
        
        Returns:
            Dict: 20个骨骼段长度（米）
            {
                'spine_base_to_spine_mid': float,
                'spine_mid_to_spine_shoulder': float,
                'spine_shoulder_to_head': float,
                'spine_shoulder_to_shoulder_left': float,
                'shoulder_left_to_elbow_left': float,
                'elbow_left_to_wrist_left': float,
                'wrist_left_to_hand_left': float,
                'spine_shoulder_to_shoulder_right': float,
                'shoulder_right_to_elbow_right': float,
                'elbow_right_to_wrist_right': float,
                'wrist_right_to_hand_right': float,
                'spine_base_to_hip_left': float,
                'hip_left_to_knee_left': float,
                'knee_left_to_ankle_left': float,
                'ankle_left_to_foot_left': float,
                'spine_base_to_hip_right': float,
                'hip_right_to_knee_right': float,
                'knee_right_to_ankle_right': float,
                'ankle_right_to_foot_right': float,
            }
        """
```

---

## 5. 稳定性检测 API

### 5.1 StabilityDetector

```python
class StabilityDetector:
    """稳定性检测器"""
    
    def __init__(self, window_size: int = 10):
        """
        初始化检测器
        
        Args:
            window_size: 滑动窗口大小
        """
    
    def add_frame(self, result: HolisticResult) -> None:
        """添加帧到检测窗口"""
    
    def get_stability(self) -> StabilityResult:
        """获取稳定性结果"""
    
    def reset(self) -> None:
        """重置检测器"""

@dataclass
class StabilityResult:
    is_stable: bool        # 是否稳定
    stable_frames: int     # 连续稳定帧数
```

---

## 6. 硬件接口 API

### 6.1 Intrinsics

```python
@dataclass
class Intrinsics:
    """相机内参"""
    width: int       # 图像宽度
    height: int      # 图像高度
    fx: float        # X方向焦距
    fy: float        # Y方向焦距
    ppx: float       # 主点X坐标
    ppy: float       # 主点Y坐标
    coeffs: List[float]  # 畸变系数
```

---

## 7. 常量定义

### 7.1 骨骼连接

```python
# src/core/constants.py

POSE_CONNECTIONS = [
    (11, 12),  # 左肩-右肩
    (11, 13),  # 左肩-左肘
    (13, 15),  # 左肘-左腕
    (12, 14),  # 右肩-右肘
    (14, 16),  # 右肘-右腕
    (11, 23),  # 左肩-左髋
    (12, 24),  # 右肩-右髋
    (23, 24),  # 左髋-右髋
    (23, 25),  # 左髋-左膝
    (25, 27),  # 左膝-左踝
    (24, 26),  # 右髋-右膝
    (26, 28),  # 右膝-右踝
    ...
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),      # 食指
    (0, 9), (9, 10), (10, 11), (11, 12), # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20), # 小指
    (5, 9), (9, 13), (13, 17),           # 掌骨连接
]
```

### 7.2 忽略的关键点

```python
# 面部细节关键点（绘制时忽略）
POSE_IGNORED_LANDMARKS = set(range(1, 11)) | set(range(17, 23))
```
