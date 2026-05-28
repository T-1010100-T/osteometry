"""
姿态估计器
封装 MediaPipe Pose 进行人体姿态检测
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..utils.mediapipe_config import ensure_mediapipe_env
ensure_mediapipe_env()

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Landmark:
    """单个关键点数据"""
    x: float            # 归一化X坐标 [0, 1]
    y: float            # 归一化Y坐标 [0, 1]  
    z: float            # 相对深度 (MediaPipe估计)
    visibility: float   # 可见性置信度 [0, 1]
    
    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """转换为像素坐标"""
        return (int(self.x * width), int(self.y * height))
    
    def is_visible(self, threshold: float = 0.5) -> bool:
        """判断是否可见"""
        return self.visibility >= threshold


@dataclass
class PoseResult:
    """姿态检测结果"""
    landmarks: List[Landmark] = field(default_factory=list)     # 33个2D关键点
    world_landmarks: List[Landmark] = field(default_factory=list)  # 世界坐标关键点
    detected: bool = False          # 是否检测到人体
    timestamp: float = 0.0          # 时间戳
    
    def get_landmark(self, index: int) -> Optional[Landmark]:
        """获取指定索引的关键点"""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None
    
    def get_visible_landmarks(self, threshold: float = 0.5) -> List[Tuple[int, Landmark]]:
        """获取所有可见的关键点"""
        return [
            (i, lm) for i, lm in enumerate(self.landmarks) 
            if lm.is_visible(threshold)
        ]
    
    def get_key_landmarks_visibility(self, key_indices: List[int], threshold: float = 0.5) -> float:
        """计算关键关键点的平均可见性"""
        if not self.landmarks or not key_indices:
            return 0.0
        
        visible_count = sum(
            1 for i in key_indices 
            if i < len(self.landmarks) and self.landmarks[i].is_visible(threshold)
        )
        return visible_count / len(key_indices)


class PoseEstimator:
    """
    姿态估计器
    
    封装 MediaPipe Pose，提供人体姿态检测功能
    
    Example:
        >>> estimator = PoseEstimator(model_complexity=2)
        >>> result = estimator.detect(rgb_image)
        >>> if result.detected:
        >>>     for lm in result.landmarks:
        >>>         print(f"({lm.x}, {lm.y})")
        >>> estimator.close()
    """
    
    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
        smooth_landmarks: bool = True
    ):
        """
        初始化姿态估计器
        
        Args:
            model_complexity: 模型复杂度 (0=Lite, 1=Full, 2=Heavy)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            enable_segmentation: 是否启用分割（会增加计算量）
            smooth_landmarks: 是否平滑关键点
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe 未安装，请运行: pip install mediapipe")
        
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self._frame_count = 0
        self._detect_count = 0
        
        logger.info(f"PoseEstimator 初始化完成 (complexity={model_complexity})")
    
    def detect(self, image: np.ndarray, timestamp: float = 0.0) -> PoseResult:
        """
        检测图像中的人体姿态
        
        Args:
            image: RGB图像，shape (H, W, 3)，注意必须是RGB格式
            timestamp: 时间戳
        
        Returns:
            PoseResult: 姿态检测结果
        """
        self._frame_count += 1
        
        # 确保是RGB格式
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.warning("输入图像格式不正确，应为 (H, W, 3)")
            return PoseResult(detected=False, timestamp=timestamp)
        
        # MediaPipe 检测
        results = self._pose.process(image)
        
        if not results.pose_landmarks:
            return PoseResult(detected=False, timestamp=timestamp)
        
        self._detect_count += 1
        
        # 转换关键点
        landmarks = self._convert_landmarks(results.pose_landmarks.landmark)
        
        # 世界坐标关键点
        world_landmarks = []
        if results.pose_world_landmarks:
            world_landmarks = self._convert_landmarks(results.pose_world_landmarks.landmark)
        
        return PoseResult(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            detected=True,
            timestamp=timestamp
        )
    
    def _convert_landmarks(self, mp_landmarks) -> List[Landmark]:
        """转换 MediaPipe 关键点格式"""
        landmarks = []
        for lm in mp_landmarks:
            landmarks.append(Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            ))
        return landmarks
    
    def detect_static(self, image: np.ndarray) -> PoseResult:
        """
        静态图像检测（每次都重新检测，不使用跟踪）
        
        Args:
            image: RGB图像
        
        Returns:
            PoseResult: 姿态检测结果
        """
        # 创建临时的静态模式检测器
        static_pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence
        )
        
        results = static_pose.process(image)
        static_pose.close()
        
        if not results.pose_landmarks:
            return PoseResult(detected=False)
        
        landmarks = self._convert_landmarks(results.pose_landmarks.landmark)
        world_landmarks = []
        if results.pose_world_landmarks:
            world_landmarks = self._convert_landmarks(results.pose_world_landmarks.landmark)
        
        return PoseResult(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            detected=True
        )
    
    def reset(self) -> None:
        """重置估计器状态（清除跟踪历史）"""
        self._pose.reset()
        logger.debug("PoseEstimator 状态已重置")
    
    def close(self) -> None:
        """释放资源"""
        if self._pose:
            self._pose.close()
            self._pose = None
        logger.info(f"PoseEstimator 关闭 (检测率: {self.detection_rate:.1%})")
    
    @property
    def frame_count(self) -> int:
        """处理的帧数"""
        return self._frame_count
    
    @property
    def detect_count(self) -> int:
        """检测成功的帧数"""
        return self._detect_count
    
    @property
    def detection_rate(self) -> float:
        """检测成功率"""
        if self._frame_count == 0:
            return 0.0
        return self._detect_count / self._frame_count
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class PoseEstimatorCompat(PoseEstimator):
    """
    兼容性姿态估计器
    
    可选使用 Holistic 后端，同时保持与原 PoseEstimator API 完全兼容
    
    Example:
        >>> # 使用原 Pose 后端
        >>> estimator = PoseEstimatorCompat(use_holistic=False)
        >>> 
        >>> # 使用 Holistic 后端（支持手部检测）
        >>> estimator = PoseEstimatorCompat(use_holistic=True)
        >>> result = estimator.detect(image)  # 返回 PoseResult
        >>> holistic_result = estimator.detect_holistic(image)  # 返回 HolisticResult
    """
    
    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
        smooth_landmarks: bool = True,
        use_holistic: bool = False,
        enable_hands: bool = True
    ):
        """
        初始化兼容性姿态估计器
        
        Args:
            model_complexity: 模型复杂度 (0=Lite, 1=Full, 2=Heavy)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            enable_segmentation: 是否启用分割
            smooth_landmarks: 是否平滑关键点
            use_holistic: 是否使用 Holistic 后端
            enable_hands: 使用 Holistic 时是否启用手部检测
        """
        self.use_holistic = use_holistic
        self._holistic_estimator = None
        
        if use_holistic:
            # 使用 Holistic 后端
            from .holistic_estimator import HolisticEstimator
            self._holistic_estimator = HolisticEstimator(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                enable_hands=enable_hands,
                smooth_landmarks=smooth_landmarks
            )
            # 设置兼容属性
            self.model_complexity = model_complexity
            self.min_detection_confidence = min_detection_confidence
            self.min_tracking_confidence = min_tracking_confidence
            self._frame_count = 0
            self._detect_count = 0
            logger.info(f"PoseEstimatorCompat 初始化完成 (Holistic后端, hands={enable_hands})")
        else:
            # 使用原 Pose 后端
            super().__init__(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                enable_segmentation=enable_segmentation,
                smooth_landmarks=smooth_landmarks
            )
    
    def detect(self, image: np.ndarray, timestamp: float = 0.0) -> PoseResult:
        """
        检测图像中的人体姿态
        
        返回与原 PoseEstimator 相同格式的结果
        
        Args:
            image: RGB图像
            timestamp: 时间戳
        
        Returns:
            PoseResult: 姿态检测结果（仅身体33点）
        """
        if self.use_holistic and self._holistic_estimator:
            result = self._holistic_estimator.detect_pose_only(image, timestamp)
            self._frame_count = self._holistic_estimator.frame_count
            self._detect_count = self._holistic_estimator.detect_count
            return result
        else:
            return super().detect(image, timestamp)
    
    def detect_holistic(self, image: np.ndarray, timestamp: float = 0.0):
        """
        检测身体和手部关键点（仅 Holistic 后端可用）
        
        Args:
            image: RGB图像
            timestamp: 时间戳
        
        Returns:
            HolisticResult: 包含身体和手部的完整结果
        
        Raises:
            RuntimeError: 如果未使用 Holistic 后端
        """
        if not self.use_holistic or not self._holistic_estimator:
            raise RuntimeError("detect_holistic 仅在 use_holistic=True 时可用")
        
        return self._holistic_estimator.detect(image, timestamp)
    
    def reset(self) -> None:
        """重置估计器状态"""
        if self.use_holistic and self._holistic_estimator:
            self._holistic_estimator.reset()
        else:
            super().reset()
    
    def close(self) -> None:
        """释放资源"""
        if self.use_holistic and self._holistic_estimator:
            self._holistic_estimator.close()
            self._holistic_estimator = None
        else:
            super().close()
    
    @property
    def frame_count(self) -> int:
        if self.use_holistic and self._holistic_estimator:
            return self._holistic_estimator.frame_count
        return self._frame_count
    
    @property
    def detect_count(self) -> int:
        if self.use_holistic and self._holistic_estimator:
            return self._holistic_estimator.detect_count
        return self._detect_count
