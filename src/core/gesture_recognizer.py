"""
手势识别器

基于手部关键点几何特征识别预定义手势

**Feature: holistic-hand-integration**
**Properties: 12, 13**
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np

from .hand_skeleton import HandSkeleton3D
from .constants import FINGER_LANDMARKS
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GestureType(Enum):
    """手势类型"""
    UNKNOWN = "unknown"
    FIST = "fist"           # 握拳
    OPEN_PALM = "open_palm" # 手掌张开
    POINTING = "pointing"   # 食指指向
    OK = "ok"               # OK手势


@dataclass
class GestureResult:
    """
    手势识别结果
    
    Attributes:
        gesture: 识别的手势类型
        confidence: 置信度 (0-1)
        hand: 左手或右手
        stable_frames: 连续识别帧数
    """
    gesture: GestureType
    confidence: float
    hand: str  # "left" / "right"
    stable_frames: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'gesture': self.gesture.value,
            'confidence': self.confidence,
            'hand': self.hand,
            'stable_frames': self.stable_frames
        }


class GestureRecognizer:
    """
    手势识别器
    
    基于手部关键点几何特征识别预定义手势
    
    **Property 12: 手势分类正确性**
    **Property 13: 手势稳定触发**
    
    Example:
        >>> recognizer = GestureRecognizer(stability_threshold=10)
        >>> result = recognizer.recognize(hand_skeleton)
        >>> triggered = recognizer.update(hand_skeleton)
        >>> if triggered:
        ...     print(f"触发手势: {triggered.value}")
    """
    
    # 默认阈值（米）
    DEFAULT_FIST_CURL_DISTANCE = 0.03      # 握拳时指尖到MCP距离
    DEFAULT_OPEN_EXTEND_DISTANCE = 0.06    # 张开时指尖到MCP距离
    DEFAULT_OK_TOUCH_DISTANCE = 0.025      # OK手势拇指食指距离
    
    def __init__(
        self,
        stability_threshold: int = 10,
        confidence_threshold: float = 0.8,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        初始化手势识别器
        
        Args:
            stability_threshold: 触发动作所需的连续帧数
            confidence_threshold: 最小置信度阈值
            thresholds: 自定义阈值字典
        """
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        
        # 阈值配置
        self.thresholds = {
            'fist_curl_distance': self.DEFAULT_FIST_CURL_DISTANCE,
            'open_extend_distance': self.DEFAULT_OPEN_EXTEND_DISTANCE,
            'ok_touch_distance': self.DEFAULT_OK_TOUCH_DISTANCE,
        }
        if thresholds:
            self.thresholds.update(thresholds)
        
        # 状态跟踪
        self._current_gesture = GestureType.UNKNOWN
        self._stable_count = 0
        self._gesture_history: deque = deque(maxlen=20)
        self._last_triggered: Optional[GestureType] = None
    
    def recognize(self, hand_skeleton: HandSkeleton3D) -> GestureResult:
        """
        识别单帧手势
        
        **Property 12: 手势分类正确性**
        
        Args:
            hand_skeleton: 手部骨骼模型
        
        Returns:
            GestureResult: 手势识别结果
        """
        # 检查各种手势
        is_fist, fist_conf = self._is_fist(hand_skeleton)
        is_open, open_conf = self._is_open_palm(hand_skeleton)
        is_pointing, point_conf = self._is_pointing(hand_skeleton)
        is_ok, ok_conf = self._is_ok(hand_skeleton)
        
        # 选择置信度最高的手势
        candidates = [
            (GestureType.FIST, is_fist, fist_conf),
            (GestureType.OPEN_PALM, is_open, open_conf),
            (GestureType.POINTING, is_pointing, point_conf),
            (GestureType.OK, is_ok, ok_conf),
        ]
        
        best_gesture = GestureType.UNKNOWN
        best_confidence = 0.0
        
        for gesture, detected, confidence in candidates:
            if detected and confidence > best_confidence:
                best_gesture = gesture
                best_confidence = confidence
        
        return GestureResult(
            gesture=best_gesture,
            confidence=best_confidence,
            hand=hand_skeleton.handedness,
            stable_frames=self._stable_count if best_gesture == self._current_gesture else 0
        )
    
    def update(self, hand_skeleton: HandSkeleton3D) -> Optional[GestureType]:
        """
        更新手势状态，返回稳定触发的手势
        
        **Property 13: 手势稳定触发**
        
        Args:
            hand_skeleton: 手部骨骼模型
        
        Returns:
            触发的手势类型，或 None
        """
        result = self.recognize(hand_skeleton)
        
        # 记录历史
        self._gesture_history.append(result)
        
        # 检查置信度
        if result.confidence < self.confidence_threshold:
            self._stable_count = 0
            self._current_gesture = GestureType.UNKNOWN
            return None
        
        # 更新稳定计数
        if result.gesture == self._current_gesture:
            self._stable_count += 1
        else:
            self._current_gesture = result.gesture
            self._stable_count = 1
        
        # 检查是否达到稳定阈值
        if self._stable_count >= self.stability_threshold:
            # 只在首次达到阈值时触发
            if self._last_triggered != self._current_gesture:
                self._last_triggered = self._current_gesture
                logger.debug(f"手势触发: {self._current_gesture.value}")
                return self._current_gesture
        
        return None
    
    def _is_fist(self, skeleton: HandSkeleton3D) -> Tuple[bool, float]:
        """
        检测握拳手势
        
        **Property 12: 手势分类正确性**
        如果所有手指的指尖到MCP距离 < 3cm → 分类为 "fist"
        
        Args:
            skeleton: 手部骨骼模型
        
        Returns:
            (是否为握拳, 置信度)
        """
        threshold = self.thresholds['fist_curl_distance']
        
        # 检查除拇指外的四指
        fingers = ['index', 'middle', 'ring', 'pinky']
        curl_count = 0
        total_confidence = 0.0
        
        for finger in fingers:
            distance = skeleton.get_tip_to_mcp_distance(finger)
            if distance > 0 and distance < threshold:
                curl_count += 1
                # 距离越小，置信度越高
                total_confidence += 1.0 - (distance / threshold)
        
        # 拇指也需要弯曲
        thumb_distance = skeleton.get_tip_to_mcp_distance('thumb')
        thumb_curled = thumb_distance > 0 and thumb_distance < threshold * 1.5
        
        if curl_count >= 4 and thumb_curled:
            confidence = total_confidence / 4
            return True, min(confidence, 1.0)
        
        return False, 0.0
    
    def _is_open_palm(self, skeleton: HandSkeleton3D) -> Tuple[bool, float]:
        """
        检测手掌张开手势
        
        **Property 12: 手势分类正确性**
        如果所有手指的指尖到MCP距离 > 6cm → 分类为 "open_palm"
        
        Args:
            skeleton: 手部骨骼模型
        
        Returns:
            (是否为张开手掌, 置信度)
        """
        threshold = self.thresholds['open_extend_distance']
        
        # 检查所有五指
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        extend_count = 0
        total_confidence = 0.0
        
        for finger in fingers:
            distance = skeleton.get_tip_to_mcp_distance(finger)
            if distance > threshold:
                extend_count += 1
                # 距离越大，置信度越高（但有上限）
                conf = min(distance / threshold, 1.5) / 1.5
                total_confidence += conf
        
        if extend_count >= 5:
            confidence = total_confidence / 5
            return True, min(confidence, 1.0)
        
        return False, 0.0
    
    def _is_pointing(self, skeleton: HandSkeleton3D) -> Tuple[bool, float]:
        """
        检测食指指向手势
        
        **Property 12: 手势分类正确性**
        如果仅食指伸展 (>6cm) 且其他手指弯曲 (<3cm) → 分类为 "pointing"
        
        Args:
            skeleton: 手部骨骼模型
        
        Returns:
            (是否为指向, 置信度)
        """
        extend_threshold = self.thresholds['open_extend_distance']
        curl_threshold = self.thresholds['fist_curl_distance']
        
        # 食指必须伸展
        index_distance = skeleton.get_tip_to_mcp_distance('index')
        index_extended = index_distance > extend_threshold
        
        if not index_extended:
            return False, 0.0
        
        # 其他手指必须弯曲
        other_fingers = ['middle', 'ring', 'pinky']
        curl_count = 0
        
        for finger in other_fingers:
            distance = skeleton.get_tip_to_mcp_distance(finger)
            if distance > 0 and distance < curl_threshold:
                curl_count += 1
        
        # 拇指可以弯曲或半伸展
        thumb_distance = skeleton.get_tip_to_mcp_distance('thumb')
        thumb_ok = thumb_distance < extend_threshold
        
        if curl_count >= 3 and thumb_ok:
            # 计算置信度
            index_conf = min(index_distance / extend_threshold, 1.5) / 1.5
            return True, index_conf
        
        return False, 0.0
    
    def _is_ok(self, skeleton: HandSkeleton3D) -> Tuple[bool, float]:
        """
        检测OK手势
        
        **Property 12: 手势分类正确性**
        如果拇指尖到食指尖距离 < 2cm 且其他手指伸展 → 分类为 "ok"
        
        Args:
            skeleton: 手部骨骼模型
        
        Returns:
            (是否为OK手势, 置信度)
        """
        touch_threshold = self.thresholds['ok_touch_distance']
        extend_threshold = self.thresholds['open_extend_distance']
        
        # 拇指和食指必须接触
        thumb_tip = skeleton.get_joint('thumb_tip')
        index_tip = skeleton.get_joint('index_finger_tip')
        
        if not thumb_tip or not index_tip:
            return False, 0.0
        
        if not thumb_tip.is_valid() or not index_tip.is_valid():
            return False, 0.0
        
        touch_distance = thumb_tip.distance_to(index_tip)
        
        if touch_distance > touch_threshold:
            return False, 0.0
        
        # 其他三指应该伸展
        other_fingers = ['middle', 'ring', 'pinky']
        extend_count = 0
        
        for finger in other_fingers:
            distance = skeleton.get_tip_to_mcp_distance(finger)
            if distance > extend_threshold * 0.7:  # 稍微放宽阈值
                extend_count += 1
        
        if extend_count >= 2:  # 至少2指伸展
            # 置信度基于接触距离
            touch_conf = 1.0 - (touch_distance / touch_threshold)
            return True, touch_conf
        
        return False, 0.0
    
    def reset(self) -> None:
        """重置状态"""
        self._current_gesture = GestureType.UNKNOWN
        self._stable_count = 0
        self._gesture_history.clear()
        self._last_triggered = None
        logger.debug("GestureRecognizer 已重置")
    
    @property
    def current_gesture(self) -> GestureType:
        """当前识别的手势"""
        return self._current_gesture
    
    @property
    def stable_count(self) -> int:
        """当前手势的稳定帧数"""
        return self._stable_count
    
    @property
    def last_triggered(self) -> Optional[GestureType]:
        """最后触发的手势"""
        return self._last_triggered
