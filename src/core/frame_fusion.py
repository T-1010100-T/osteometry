"""
多帧融合器

将多帧测量数据融合为单一高质量结果，支持IQR异常值检测和质量加权平均

**Feature: smart-data-collector**
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import numpy as np

from .smart_collector_types import FusionResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameData:
    """单帧数据"""
    body_data: Dict = field(default_factory=dict)
    left_hand_data: Dict = field(default_factory=dict)
    right_hand_data: Dict = field(default_factory=dict)
    quality_score: float = 0.0
    timestamp: float = 0.0


class FrameFusion:
    """
    多帧融合器
    
    通过IQR方法检测并移除异常值，使用质量加权平均融合多帧数据
    
    Example:
        >>> fusion = FrameFusion(fusion_frames=5)
        >>> for measurement in measurements:
        ...     if fusion.add_frame(measurement, quality):
        ...         result = fusion.fuse()
        ...         break
    """
    
    def __init__(
        self,
        fusion_frames: int = 5,
        iqr_multiplier: float = 1.5,
        min_frames: int = 3
    ):
        """
        初始化多帧融合器
        
        Args:
            fusion_frames: 融合帧数
            iqr_multiplier: IQR倍数，用于异常值检测
            min_frames: 最小有效帧数
        """
        self.fusion_frames = fusion_frames
        self.iqr_multiplier = iqr_multiplier
        self.min_frames = min_frames
        
        self._buffer: List[FrameData] = []
        self._outliers_removed = 0
        
        logger.debug(f"FrameFusion 初始化: frames={fusion_frames}, iqr={iqr_multiplier}")

    
    def add_frame(
        self,
        body_data: Optional[Dict] = None,
        left_hand_data: Optional[Dict] = None,
        right_hand_data: Optional[Dict] = None,
        quality_score: float = 0.5
    ) -> bool:
        """
        添加帧到融合缓冲区
        
        Args:
            body_data: 身体测量数据字典
            left_hand_data: 左手测量数据字典
            right_hand_data: 右手测量数据字典
            quality_score: 质量分数
        
        Returns:
            缓冲区是否已满
        """
        frame = FrameData(
            body_data=body_data or {},
            left_hand_data=left_hand_data or {},
            right_hand_data=right_hand_data or {},
            quality_score=quality_score,
            timestamp=time.time()
        )
        
        self._buffer.append(frame)
        
        return len(self._buffer) >= self.fusion_frames
    
    def add_values(self, values: List[float], quality_scores: List[float]) -> bool:
        """
        直接添加数值列表（用于测试）
        
        Args:
            values: 测量值列表
            quality_scores: 对应的质量分数列表
        """
        for val, q in zip(values, quality_scores):
            self._buffer.append(FrameData(
                body_data={'value': val},
                quality_score=q,
                timestamp=time.time()
            ))
        return len(self._buffer) >= self.fusion_frames
    
    def fuse(self) -> FusionResult:
        """
        执行多帧融合
        
        Returns:
            FusionResult: 融合结果
        """
        if len(self._buffer) < self.min_frames:
            logger.warning(f"帧数不足: {len(self._buffer)} < {self.min_frames}")
            return FusionResult(frames_used=0)
        
        self._outliers_removed = 0
        
        # 融合身体数据
        body_result = self._fuse_data_dict(
            [f.body_data for f in self._buffer],
            [f.quality_score for f in self._buffer]
        )
        
        # 融合左手数据
        left_hand_result = self._fuse_data_dict(
            [f.left_hand_data for f in self._buffer],
            [f.quality_score for f in self._buffer]
        )
        
        # 融合右手数据
        right_hand_result = self._fuse_data_dict(
            [f.right_hand_data for f in self._buffer],
            [f.quality_score for f in self._buffer]
        )
        
        # 如果融合结果为空，尝试使用最高质量帧的原始数据
        if not body_result:
            best_frame = max(self._buffer, key=lambda f: f.quality_score)
            if best_frame.body_data:
                body_result = best_frame.body_data.copy()
                logger.debug("使用最高质量帧的身体数据")
        
        if not left_hand_result:
            best_frame = max(
                (f for f in self._buffer if f.left_hand_data),
                key=lambda f: f.quality_score,
                default=None
            )
            if best_frame and best_frame.left_hand_data:
                left_hand_result = best_frame.left_hand_data.copy()
                logger.debug("使用最高质量帧的左手数据")
        
        if not right_hand_result:
            best_frame = max(
                (f for f in self._buffer if f.right_hand_data),
                key=lambda f: f.quality_score,
                default=None
            )
            if best_frame and best_frame.right_hand_data:
                right_hand_result = best_frame.right_hand_data.copy()
                logger.debug("使用最高质量帧的右手数据")
        
        # 计算融合后置信度
        avg_quality = np.mean([f.quality_score for f in self._buffer])
        
        return FusionResult(
            body_measurement=body_result if body_result else None,
            left_hand=left_hand_result if left_hand_result else None,
            right_hand=right_hand_result if right_hand_result else None,
            confidence=float(avg_quality),
            frames_used=len(self._buffer),
            outliers_removed=self._outliers_removed,
            fusion_timestamp=time.time()
        )
    
    def fuse_values(self, values: List[float], weights: List[float]) -> Tuple[float, int]:
        """
        直接融合数值列表（用于测试）
        
        Args:
            values: 测量值列表
            weights: 权重列表
        
        Returns:
            (融合结果, 移除的异常值数量)
        """
        if len(values) < self.min_frames:
            return 0.0, 0
        
        # IQR 异常值检测
        clean_values, clean_weights, outliers = self._remove_outliers_iqr(values, weights)
        
        if len(clean_values) == 0:
            return 0.0, len(values)
        
        # 质量加权平均
        result = self._weighted_average(clean_values, clean_weights)
        
        return result, outliers
    
    def _fuse_data_dict(
        self,
        data_list: List[Dict],
        quality_scores: List[float]
    ) -> Dict:
        """融合数据字典列表"""
        if not data_list or not any(data_list):
            return {}
        
        # 收集所有键
        all_keys = set()
        for data in data_list:
            if data:
                all_keys.update(data.keys())
        
        result = {}
        for key in all_keys:
            # 收集该键的所有值
            values = []
            weights = []
            non_numeric_values = []
            
            for data, quality in zip(data_list, quality_scores):
                if data and key in data:
                    val = data[key]
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        if not np.isnan(val):
                            values.append(float(val))
                            weights.append(quality)
                    elif isinstance(val, bool):
                        non_numeric_values.append(val)
                    elif isinstance(val, dict):
                        # 保留最后一个有效的字典值
                        if val:
                            result[key] = val
            
            if values:
                if len(values) >= self.min_frames:
                    # IQR 异常值检测
                    clean_values, clean_weights, outliers = self._remove_outliers_iqr(values, weights)
                    self._outliers_removed += outliers
                    
                    if clean_values:
                        # 质量加权平均
                        result[key] = self._weighted_average(clean_values, clean_weights)
                else:
                    # 数据不足，直接平均
                    result[key] = np.mean(values)
            elif non_numeric_values:
                # 布尔值取多数
                result[key] = sum(non_numeric_values) > len(non_numeric_values) / 2
        
        return result
    
    def _remove_outliers_iqr(
        self,
        values: List[float],
        weights: List[float]
    ) -> Tuple[List[float], List[float], int]:
        """
        使用IQR方法移除异常值
        
        Args:
            values: 数值列表
            weights: 权重列表
        
        Returns:
            (清洗后的值, 清洗后的权重, 移除数量)
        """
        if len(values) < 4:  # IQR需要足够数据
            return values, weights, 0
        
        arr = np.array(values)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        clean_values = []
        clean_weights = []
        outliers = 0
        
        for val, weight in zip(values, weights):
            if lower_bound <= val <= upper_bound:
                clean_values.append(val)
                clean_weights.append(weight)
            else:
                outliers += 1
        
        return clean_values, clean_weights, outliers
    
    def _weighted_average(self, values: List[float], weights: List[float]) -> float:
        """
        计算质量加权平均
        
        Args:
            values: 数值列表
            weights: 权重列表
        
        Returns:
            加权平均值
        """
        if not values:
            return 0.0
        
        total_weight = sum(weights)
        if total_weight == 0:
            return np.mean(values)
        
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return weighted_sum / total_weight
    
    def reset(self) -> None:
        """重置融合器"""
        self._buffer.clear()
        self._outliers_removed = 0
        logger.debug("FrameFusion 已重置")
    
    @property
    def buffer_size(self) -> int:
        """当前缓冲区大小"""
        return len(self._buffer)
    
    @property
    def is_ready(self) -> bool:
        """是否准备好融合"""
        return len(self._buffer) >= self.fusion_frames
