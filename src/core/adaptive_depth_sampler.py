"""
自适应深度采样器模块

基于身体部位的自适应深度采样，提供鲁棒的深度值获取
"""
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.depth_config import (
    SamplerConfig,
    get_sampling_strategy,
    KEYPOINT_ID_TO_BODY_PART,
)


class AdaptiveDepthSampler:
    """
    自适应深度采样器
    
    功能：
    - 基于身体部位的自适应窗口大小
    - 前景-背景分离
    - 多种统计方法（中位数、截尾均值等）
    - 时序一致性检查
    """
    
    def __init__(self, config: Optional[SamplerConfig] = None):
        """
        初始化采样器
        
        Args:
            config: 采样器配置
        """
        self.config = config or SamplerConfig()
        
        # 历史数据缓冲区（用于时序一致性检查）
        self.history_buffer: Dict[int, deque] = {}
    
    def sample(
        self,
        depth_map: np.ndarray,
        x: int,
        y: int,
        keypoint_id: int,
        confidence: float,
        depth_scale: float = 1.0
    ) -> float:
        """
        采样单个关键点的深度值
        
        Args:
            depth_map: 深度图 (H, W), uint16 或 float32
            x, y: 像素坐标
            keypoint_id: MediaPipe 关键点 ID (0-32)
            confidence: 关键点置信度 (0-1)
            depth_scale: 深度比例因子（将原始值转换为米）
        
        Returns:
            深度值（米），无效返回 0.0
        """
        h, w = depth_map.shape[:2]
        
        # 边界检查
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0.0
        
        # 获取采样策略
        window_size, fg_ratio, method, outlier_thresh, max_change = get_sampling_strategy(keypoint_id)
        
        # 根据置信度动态调整策略
        if confidence < 0.3:
            # 极低置信度：更保守的策略
            window_size = max(3, window_size - 4)
            fg_ratio = min(0.9, fg_ratio + 0.2)
            outlier_thresh *= 0.5
        elif confidence < 0.5:
            # 低置信度：适度调整
            window_size = max(3, window_size - 2)
            fg_ratio = min(0.85, fg_ratio + 0.1)
        
        # 提取 ROI 区域
        roi, valid_mask = self._extract_roi(depth_map, x, y, window_size)
        
        if roi.size == 0 or not np.any(valid_mask):
            return 0.0
        
        # 转换为米
        if depth_map.dtype == np.uint16:
            roi = roi.astype(np.float32) * depth_scale
        
        # 三阶段采样
        depth_value = self._three_stage_sampling(
            roi, valid_mask, fg_ratio, method, outlier_thresh
        )
        
        # 时序一致性检查
        if self.config.enable_temporal_check and depth_value > 0:
            depth_value = self._temporal_consistency_check(
                keypoint_id, depth_value, max_change
            )
        
        return depth_value
    
    def batch_sample(
        self,
        depth_map: np.ndarray,
        keypoints: List[Tuple[int, int]],
        keypoint_ids: List[int],
        confidences: List[float],
        depth_scale: float = 1.0
    ) -> List[float]:
        """
        批量采样关键点深度
        
        Args:
            depth_map: 深度图
            keypoints: 像素坐标列表 [(x, y), ...]
            keypoint_ids: 关键点 ID 列表
            confidences: 置信度列表
            depth_scale: 深度比例因子
        
        Returns:
            深度值列表（米）
        """
        n_points = len(keypoints)
        depths = []
        
        for i in range(n_points):
            x, y = keypoints[i]
            kp_id = keypoint_ids[i] if i < len(keypoint_ids) else i
            conf = confidences[i] if i < len(confidences) else 0.5
            
            depth = self.sample(depth_map, int(x), int(y), kp_id, conf, depth_scale)
            depths.append(depth)
        
        return depths
    
    def _extract_roi(
        self,
        depth_map: np.ndarray,
        center_x: int,
        center_y: int,
        window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        安全地提取 ROI 区域
        
        Returns:
            (roi, valid_mask)
        """
        h, w = depth_map.shape[:2]
        half = window_size // 2
        
        # 计算边界
        x_start = max(0, center_x - half)
        x_end = min(w, center_x + half + 1)
        y_start = max(0, center_y - half)
        y_end = min(h, center_y + half + 1)
        
        # 提取 ROI
        roi = depth_map[y_start:y_end, x_start:x_end].copy()
        
        # 创建有效掩码（深度 > 0 且非 NaN）
        if roi.dtype == np.float32 or roi.dtype == np.float64:
            valid_mask = (roi > 0) & (~np.isnan(roi))
        else:
            valid_mask = roi > 0
        
        return roi, valid_mask
    
    def _three_stage_sampling(
        self,
        roi: np.ndarray,
        valid_mask: np.ndarray,
        fg_ratio: float,
        method: str,
        outlier_thresh: float
    ) -> float:
        """
        三阶段深度采样
        
        阶段1: 提取有效深度值
        阶段2: 前景-背景分离
        阶段3: 统计聚合
        """
        # 阶段1: 提取有效深度值
        valid_depths = roi[valid_mask]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # 阶段2: 前景-背景分离
        # 假设前景深度值较小（离相机更近）
        sorted_depths = np.sort(valid_depths)
        
        # 确定前景样本数量
        n_samples = max(3, int(len(sorted_depths) * fg_ratio))
        n_samples = min(n_samples, len(sorted_depths))
        
        foreground_depths = sorted_depths[:n_samples]
        
        # 阶段3: 统计聚合
        if method == 'median':
            return float(np.median(foreground_depths))
        elif method == 'mean':
            return float(np.mean(foreground_depths))
        elif method == 'trimmed_mean':
            return self._trimmed_mean(foreground_depths, trim=0.1)
        elif method == 'robust_mean':
            return self._robust_mean(foreground_depths, outlier_thresh)
        elif method == 'mode':
            return self._find_mode(foreground_depths)
        else:
            return float(np.median(foreground_depths))
    
    def _trimmed_mean(self, depths: np.ndarray, trim: float = 0.1) -> float:
        """截尾均值（去除极端值）"""
        if len(depths) == 0:
            return 0.0
        
        n_trim = int(len(depths) * trim)
        if n_trim == 0 or len(depths) <= 2 * n_trim:
            return float(np.mean(depths))
        
        trimmed = depths[n_trim:-n_trim]
        return float(np.mean(trimmed))
    
    def _robust_mean(self, depths: np.ndarray, outlier_thresh: float) -> float:
        """鲁棒均值（基于 IQR 去除离群值）"""
        if len(depths) < 4:
            return float(np.mean(depths))
        
        # 计算四分位数
        q1 = np.percentile(depths, 25)
        q3 = np.percentile(depths, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return float(np.mean(depths))
        
        # 定义离群值边界
        # 使用 outlier_thresh 作为 IQR 的倍数
        multiplier = outlier_thresh * 100  # 转换为合理的倍数
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        # 过滤离群值
        filtered = depths[(depths >= lower_bound) & (depths <= upper_bound)]
        
        if len(filtered) == 0:
            return float(np.mean(depths))
        
        return float(np.mean(filtered))
    
    def _find_mode(self, depths: np.ndarray, bins: int = 20) -> float:
        """寻找深度的众数"""
        if len(depths) == 0:
            return 0.0
        
        # 计算直方图
        hist, bin_edges = np.histogram(depths, bins=bins)
        
        # 找到最大频率的 bin
        max_bin_idx = np.argmax(hist)
        
        # 返回 bin 的中值
        mode_value = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
        
        return float(mode_value)
    
    def _temporal_consistency_check(
        self,
        keypoint_id: int,
        current_depth: float,
        max_change: float
    ) -> float:
        """
        时序一致性检查
        
        如果当前深度值与历史值差异过大，使用历史中位数
        """
        if keypoint_id not in self.history_buffer:
            self.history_buffer[keypoint_id] = deque(maxlen=self.config.history_size)
        
        buffer = self.history_buffer[keypoint_id]
        
        # 如果历史数据不足，直接添加并返回
        if len(buffer) < 2:
            buffer.append(current_depth)
            return current_depth
        
        # 计算历史中位数
        history_median = float(np.median(list(buffer)))
        
        # 检查变化是否合理
        if history_median > 0:
            change_ratio = abs(current_depth - history_median) / history_median
            
            if change_ratio > max_change:
                # 变化太大，使用历史中位数
                buffer.append(history_median)
                return history_median
        
        # 变化合理，添加当前值
        buffer.append(current_depth)
        return current_depth
    
    def reset(self, keypoint_id: Optional[int] = None):
        """
        重置历史缓冲区
        
        Args:
            keypoint_id: 指定关键点 ID，None 表示重置所有
        """
        if keypoint_id is None:
            self.history_buffer.clear()
        elif keypoint_id in self.history_buffer:
            self.history_buffer[keypoint_id].clear()
