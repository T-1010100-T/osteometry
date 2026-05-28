"""
稳定性检测器

通过分析连续帧的关键点位移判断用户是否处于稳定状态

**Feature: smart-data-collector**
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
"""
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .smart_collector_types import StabilityResult
from .hand_result import HolisticResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StabilityDetector:
    """
    稳定性检测器
    
    通过计算连续帧关键点的移动标准差来判断用户是否稳定
    支持身体和手部使用不同的稳定阈值
    
    Example:
        >>> detector = StabilityDetector(window_size=10)
        >>> detector.add_frame(holistic_result)
        >>> result = detector.get_stability()
        >>> if result.is_stable:
        ...     print("用户已稳定")
    """
    
    def __init__(
        self,
        window_size: int = 10,
        body_threshold: float = 0.02,
        hand_threshold: float = 0.01,
        min_confidence: float = 0.5
    ):
        """
        初始化稳定性检测器
        
        Args:
            window_size: 检测窗口大小（帧数）
            body_threshold: 身体稳定阈值（米），移动标准差低于此值视为稳定
            hand_threshold: 手部稳定阈值（米）
            min_confidence: 最小置信度，低于此值的关键点不参与计算
        """
        self.window_size = window_size
        self.body_threshold = body_threshold
        self.hand_threshold = hand_threshold
        self.min_confidence = min_confidence
        
        # 滑动窗口缓冲区
        self._body_buffer: deque = deque(maxlen=window_size)
        self._left_hand_buffer: deque = deque(maxlen=window_size)
        self._right_hand_buffer: deque = deque(maxlen=window_size)
        
        # 连续稳定帧计数
        self._stable_count = 0

        
        logger.debug(f"StabilityDetector 初始化: window={window_size}, body_th={body_threshold}, hand_th={hand_threshold}")
    
    def add_frame(self, holistic_result: HolisticResult) -> None:
        """
        添加新帧到检测窗口
        
        Args:
            holistic_result: Holistic 检测结果
        """
        # 提取身体关键点坐标
        if holistic_result.pose.detected and holistic_result.pose.landmarks:
            body_points = self._extract_body_points(holistic_result)
            self._body_buffer.append(body_points)
        
        # 提取左手关键点坐标
        if holistic_result.left_hand.detected and holistic_result.left_hand.landmarks:
            left_points = self._extract_hand_points(holistic_result.left_hand.landmarks)
            self._left_hand_buffer.append(left_points)
        
        # 提取右手关键点坐标
        if holistic_result.right_hand.detected and holistic_result.right_hand.landmarks:
            right_points = self._extract_hand_points(holistic_result.right_hand.landmarks)
            self._right_hand_buffer.append(right_points)
    
    def add_points(
        self,
        body_points: Optional[np.ndarray] = None,
        left_hand_points: Optional[np.ndarray] = None,
        right_hand_points: Optional[np.ndarray] = None
    ) -> None:
        """
        直接添加关键点坐标（用于测试）
        
        Args:
            body_points: 身体关键点 shape (N, 3)
            left_hand_points: 左手关键点 shape (21, 3)
            right_hand_points: 右手关键点 shape (21, 3)
        """
        if body_points is not None:
            self._body_buffer.append(body_points)
        if left_hand_points is not None:
            self._left_hand_buffer.append(left_hand_points)
        if right_hand_points is not None:
            self._right_hand_buffer.append(right_hand_points)
    
    def get_stability(self) -> StabilityResult:
        """
        获取当前稳定性状态
        
        Returns:
            StabilityResult: 稳定性检测结果
        """
        # 计算各部位的移动量
        body_movement = self._calculate_movement(self._body_buffer)
        left_hand_movement = self._calculate_movement(self._left_hand_buffer)
        right_hand_movement = self._calculate_movement(self._right_hand_buffer)
        
        # 判断各部位是否稳定
        body_stable = body_movement < self.body_threshold if body_movement >= 0 else True
        left_hand_stable = left_hand_movement < self.hand_threshold if left_hand_movement >= 0 else True
        right_hand_stable = right_hand_movement < self.hand_threshold if right_hand_movement >= 0 else True
        
        # 综合手部移动量
        hand_movement = max(
            left_hand_movement if left_hand_movement >= 0 else 0,
            right_hand_movement if right_hand_movement >= 0 else 0
        )
        
        # 判断整体是否稳定（需要所有检测到的部位都稳定）
        is_stable = body_stable and left_hand_stable and right_hand_stable
        
        # 更新稳定计数
        if is_stable and len(self._body_buffer) >= self.window_size:
            self._stable_count += 1
        else:
            self._stable_count = 0
        
        # 计算稳定进度
        progress = self._calculate_progress(body_movement, hand_movement)
        
        return StabilityResult(
            is_stable=is_stable and len(self._body_buffer) >= self.window_size,
            progress=progress,
            body_stable=body_stable,
            left_hand_stable=left_hand_stable,
            right_hand_stable=right_hand_stable,
            body_movement=max(0, body_movement),
            hand_movement=hand_movement,
            stable_frames=self._stable_count
        )
    
    def _extract_body_points(self, holistic_result: HolisticResult) -> np.ndarray:
        """提取身体关键点坐标"""
        landmarks = holistic_result.pose.landmarks
        points = []
        for lm in landmarks:
            if lm.visibility >= self.min_confidence:
                points.append([lm.x, lm.y, lm.z])
        return np.array(points) if points else np.array([]).reshape(0, 3)
    
    def _extract_hand_points(self, landmarks) -> np.ndarray:
        """提取手部关键点坐标"""
        points = []
        for lm in landmarks:
            points.append([lm.x, lm.y, lm.z])
        return np.array(points) if points else np.array([]).reshape(0, 3)
    
    def _calculate_movement(self, buffer: deque) -> float:
        """
        计算关键点移动的标准差
        
        Args:
            buffer: 关键点缓冲区
        
        Returns:
            移动标准差，如果数据不足返回 -1
        """
        if len(buffer) < 2:
            return -1.0
        
        # 转换为数组 shape: (frames, points, 3)
        try:
            frames = [f for f in buffer if len(f) > 0]
            if len(frames) < 2:
                return -1.0
            
            # 找到最小点数（处理不同帧点数不同的情况）
            min_points = min(len(f) for f in frames)
            if min_points == 0:
                return -1.0
            
            # 截取相同数量的点
            aligned_frames = np.array([f[:min_points] for f in frames])
            
            # 计算每个点在时间维度上的标准差，然后取平均
            # shape: (frames, points, 3) -> (points, 3) std -> scalar mean
            std_per_point = np.std(aligned_frames, axis=0)  # (points, 3)
            movement = np.mean(std_per_point)
            
            return float(movement)
        except Exception as e:
            logger.debug(f"计算移动量失败: {e}")
            return -1.0
    
    def _calculate_progress(self, body_movement: float, hand_movement: float) -> float:
        """
        计算稳定进度 (0.0 - 1.0)
        
        基于当前移动量与阈值的比值计算
        """
        # 缓冲区填充进度
        buffer_progress = len(self._body_buffer) / self.window_size
        
        if body_movement < 0:
            return buffer_progress * 0.5  # 数据不足时最多50%
        
        # 移动量越小，进度越高
        body_ratio = max(0, 1 - body_movement / self.body_threshold)
        hand_ratio = max(0, 1 - hand_movement / self.hand_threshold) if hand_movement > 0 else 1.0
        
        # 综合进度
        stability_progress = (body_ratio * 0.7 + hand_ratio * 0.3)
        
        # 结合缓冲区填充进度
        progress = min(1.0, buffer_progress * stability_progress)
        
        return max(0.0, min(1.0, progress))
    
    def reset(self) -> None:
        """重置检测器状态"""
        self._body_buffer.clear()
        self._left_hand_buffer.clear()
        self._right_hand_buffer.clear()
        self._stable_count = 0
        logger.debug("StabilityDetector 已重置")
    
    @property
    def buffer_size(self) -> int:
        """当前缓冲区大小"""
        return len(self._body_buffer)
    
    @property
    def is_buffer_full(self) -> bool:
        """缓冲区是否已满"""
        return len(self._body_buffer) >= self.window_size
