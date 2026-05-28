"""
Holistic 估计器
封装 MediaPipe Holistic 进行身体姿态和手部检测
"""
from typing import List, Optional
import time

import numpy as np

from ..utils.mediapipe_config import ensure_mediapipe_env
ensure_mediapipe_env()

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from .pose_estimator import Landmark, PoseResult
from .hand_result import HandResult, HolisticResult
from .hand_exceptions import (
    HandDetectionError, HandModelLoadError, HandTrackingLostError, GPUMemoryError
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HolisticEstimator:
    """
    MediaPipe Holistic 估计器
    
    同时检测身体姿态和双手关键点
    
    Example:
        >>> estimator = HolisticEstimator(model_complexity=1)
        >>> result = estimator.detect(rgb_image)
        >>> if result.body_detected:
        >>>     print(f"身体检测到，左手: {result.left_hand.detected}")
        >>> estimator.close()
    """
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        enable_hands: bool = True,
        smooth_landmarks: bool = True,
        refine_face_landmarks: bool = False
    ):
        """
        初始化 Holistic 估计器
        
        Args:
            model_complexity: 模型复杂度 (0=Lite, 1=Full, 2=Heavy)
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
            enable_hands: 是否启用手部检测
            smooth_landmarks: 是否平滑关键点
            refine_face_landmarks: 是否精细化面部关键点
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe 未安装，请运行: pip install mediapipe")
        
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_hands = enable_hands
        
        # 错误处理状态
        self._fallback_mode = False  # 是否处于降级模式
        self._consecutive_hand_errors = 0
        self._max_hand_errors = 10  # 连续错误阈值
        
        # 尝试加载模型
        try:
            self._mp_holistic = mp.solutions.holistic
            self._holistic = self._mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                enable_segmentation=False,
                smooth_segmentation=False,
                refine_face_landmarks=refine_face_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        except Exception as e:
            logger.warning(f"Holistic 模型加载失败，回退到 Pose-only 模式: {e}")
            self._fallback_mode = True
            self.enable_hands = False
            self._mp_holistic = None
            
            try:
                self._holistic = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=model_complexity,
                    smooth_landmarks=smooth_landmarks,
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                logger.info("Pose-only 回退成功")
            except Exception as e2:
                logger.warning(f"Pose 模型也加载失败，使用空估算器: {e2}")
                self._fallback_mode = True
                self.enable_hands = False
                self._mp_holistic = None
                self._holistic = None
        
        self._frame_count = 0
        self._detect_count = 0
        self._hand_detection_interval = 1
        self._last_hand_detection_frame = 0
        
        # 性能监控
        self._last_latency = 0.0
        self._latency_threshold = 120  # ms
        self._adaptive_interval = False
        
        mode_str = "Pose-only (降级)" if self._fallback_mode else f"Holistic (hands={enable_hands})"
        logger.info(f"HolisticEstimator 初始化完成 (complexity={model_complexity}, mode={mode_str})")

    
    def detect(self, image: np.ndarray, timestamp: float = 0.0) -> HolisticResult:
        """
        检测图像中的身体姿态和手部关键点
        
        Args:
            image: RGB图像，shape (H, W, 3)
            timestamp: 时间戳
        
        Returns:
            HolisticResult: 包含身体和手部检测结果
        """
        start_time = time.time()
        self._frame_count += 1
        
        # 输入验证
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.warning("输入图像格式不正确，应为 (H, W, 3)")
            return HolisticResult(timestamp=timestamp)
        
        if self._holistic is None:
            return HolisticResult(timestamp=timestamp)
        
        # MediaPipe Holistic 检测
        results = self._holistic.process(image)
        
        # 解析身体关键点
        pose_result = self._parse_pose_landmarks(results, timestamp)
        
        # 解析手部关键点
        left_hand = HandResult.empty("left")
        right_hand = HandResult.empty("right")
        
        if self.enable_hands and self._should_detect_hands() and not self._fallback_mode:
            try:
                # Requirements: 12.2 - 手部检测异常时捕获并继续
                left_hand = self._parse_hand_landmarks(results.left_hand_landmarks, "left")
                right_hand = self._parse_hand_landmarks(results.right_hand_landmarks, "right")
                self._last_hand_detection_frame = self._frame_count
                self._consecutive_hand_errors = 0  # 重置错误计数
            except Exception as e:
                logger.warning(f"手部检测异常，返回空手部结果: {e}")
                self._consecutive_hand_errors += 1
                
                # 连续错误过多时临时禁用手部检测
                if self._consecutive_hand_errors >= self._max_hand_errors:
                    logger.warning(f"连续 {self._max_hand_errors} 次手部检测错误，临时禁用手部检测")
                    self._hand_detection_interval = 10  # 降低检测频率
        
        if pose_result.detected:
            self._detect_count += 1
        
        # 更新延迟
        self._last_latency = (time.time() - start_time) * 1000
        
        # 自适应调整检测间隔
        if self._adaptive_interval:
            self._adjust_detection_interval()
        
        return HolisticResult(
            pose=pose_result,
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp=timestamp
        )
    
    def detect_pose_only(self, image: np.ndarray, timestamp: float = 0.0) -> PoseResult:
        """
        仅检测身体关键点（兼容模式）
        
        返回与原 PoseEstimator 相同格式
        
        Args:
            image: RGB图像
            timestamp: 时间戳
        
        Returns:
            PoseResult: 仅包含身体关键点的结果
        """
        result = self.detect(image, timestamp)
        return result.to_pose_result()
    
    def _parse_pose_landmarks(self, results, timestamp: float) -> PoseResult:
        """解析身体关键点"""
        if not results.pose_landmarks:
            return PoseResult(detected=False, timestamp=timestamp)
        
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append(Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            ))
        
        # 世界坐标关键点
        world_landmarks = []
        if results.pose_world_landmarks:
            for lm in results.pose_world_landmarks.landmark:
                world_landmarks.append(Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility
                ))
        
        return PoseResult(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            detected=True,
            timestamp=timestamp
        )
    
    def _parse_hand_landmarks(self, hand_landmarks, handedness: str) -> HandResult:
        """解析手部关键点"""
        if not hand_landmarks:
            return HandResult.empty(handedness)
        
        landmarks = []
        total_visibility = 0.0
        
        for lm in hand_landmarks.landmark:
            landmarks.append(Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=getattr(lm, 'visibility', 1.0)
            ))
            total_visibility += getattr(lm, 'visibility', 1.0)
        
        confidence = total_visibility / len(landmarks) if landmarks else 0.0
        
        return HandResult(
            landmarks=landmarks,
            detected=True,
            handedness=handedness,
            confidence=confidence
        )
    
    def _should_detect_hands(self) -> bool:
        """判断当前帧是否应该检测手部"""
        if self._hand_detection_interval <= 1:
            return True
        return (self._frame_count - self._last_hand_detection_frame) >= self._hand_detection_interval
    
    def _adjust_detection_interval(self) -> None:
        """根据延迟自适应调整检测间隔"""
        if self._last_latency > self._latency_threshold:
            # 延迟过高，增加间隔
            self._hand_detection_interval = min(self._hand_detection_interval + 1, 5)
            logger.debug(f"延迟 {self._last_latency:.1f}ms > {self._latency_threshold}ms，"
                        f"检测间隔调整为 {self._hand_detection_interval}")
        elif self._last_latency < self._latency_threshold * 0.7 and self._hand_detection_interval > 1:
            # 延迟较低，减少间隔
            self._hand_detection_interval = max(self._hand_detection_interval - 1, 1)
    
    def set_hand_detection_interval(self, interval: int) -> None:
        """
        设置手部检测间隔
        
        Args:
            interval: 每N帧检测一次手部 (1=每帧检测)
        """
        self._hand_detection_interval = max(1, interval)
        logger.info(f"手部检测间隔设置为 {self._hand_detection_interval}")
    
    def set_adaptive_interval(self, enabled: bool, latency_threshold: float = 120) -> None:
        """
        设置自适应检测间隔
        
        Args:
            enabled: 是否启用
            latency_threshold: 延迟阈值 (ms)
        """
        self._adaptive_interval = enabled
        self._latency_threshold = latency_threshold
    
    def reset(self) -> None:
        """重置估计器状态"""
        self._holistic.reset()
        self._last_hand_detection_frame = 0
        logger.debug("HolisticEstimator 状态已重置")
    
    def close(self) -> None:
        """释放资源"""
        if self._holistic:
            self._holistic.close()
            self._holistic = None
        logger.info(f"HolisticEstimator 关闭 (检测率: {self.detection_rate:.1%})")
    
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
    
    @property
    def last_latency(self) -> float:
        """最近一帧的处理延迟 (ms)"""
        return self._last_latency
    
    @property
    def hand_detection_interval(self) -> int:
        """当前手部检测间隔"""
        return self._hand_detection_interval
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
