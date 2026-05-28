"""
深度处理器模块

深度图增强和后处理
"""
from typing import Optional, Tuple, List

import cv2
import numpy as np

from src.core.depth_config import DepthProcessorConfig, SamplerConfig
from src.core.adaptive_depth_sampler import AdaptiveDepthSampler


class DepthProcessor:
    """
    深度数据处理器
    
    功能：
    - 深度图增强（无效值填充、边缘保持滤波）
    - 自适应深度采样
    """
    
    def __init__(self, config: Optional[DepthProcessorConfig] = None):
        """
        初始化深度处理器
        
        Args:
            config: 处理器配置
        """
        self.config = config or DepthProcessorConfig()
        
        # 创建自适应采样器
        self.sampler = AdaptiveDepthSampler(self.config.sampler)
    
    def process(
        self,
        depth_frame: np.ndarray,
        depth_scale: float = 0.001
    ) -> np.ndarray:
        """
        处理深度帧
        
        Args:
            depth_frame: 原始深度帧 (H, W), uint16 或 float32
            depth_scale: 深度比例因子（将原始值转换为米）
        
        Returns:
            增强后的深度帧 (H, W), float32, 单位：米
        """
        # 转换为浮点数（米）
        if depth_frame.dtype == np.uint16:
            depth_m = depth_frame.astype(np.float32) * depth_scale
        else:
            depth_m = depth_frame.astype(np.float32)
        
        if not self.config.enable_enhancement:
            return depth_m
        
        # 1. 无效值填充
        if self.config.enable_hole_filling:
            depth_m = self._fill_invalid_values(depth_m)
        
        # 2. 边缘保持滤波
        depth_m = self._edge_preserving_filter(depth_m)
        
        return depth_m
    
    def sample_keypoint_depth(
        self,
        depth_frame: np.ndarray,
        x: int,
        y: int,
        keypoint_id: int,
        confidence: float,
        depth_scale: float = 0.001
    ) -> float:
        """
        采样关键点深度
        
        Args:
            depth_frame: 深度帧
            x, y: 像素坐标
            keypoint_id: MediaPipe 关键点 ID (0-32)
            confidence: 关键点置信度 (0-1)
            depth_scale: 深度比例因子
        
        Returns:
            深度值（米），无效返回 0.0
        """
        return self.sampler.sample(
            depth_frame, x, y, keypoint_id, confidence, depth_scale
        )
    
    def batch_sample(
        self,
        depth_frame: np.ndarray,
        keypoints: List[Tuple[int, int]],
        keypoint_ids: List[int],
        confidences: List[float],
        depth_scale: float = 0.001
    ) -> List[float]:
        """
        批量采样关键点深度
        
        Returns:
            深度值列表（米）
        """
        return self.sampler.batch_sample(
            depth_frame, keypoints, keypoint_ids, confidences, depth_scale
        )
    
    def _fill_invalid_values(self, depth_m: np.ndarray) -> np.ndarray:
        """
        填充无效深度值（0 或 NaN）
        
        使用最近有效值填充
        """
        # 创建有效掩码
        valid_mask = (depth_m > 0) & (~np.isnan(depth_m))
        
        if np.all(valid_mask):
            return depth_m
        
        if not np.any(valid_mask):
            return depth_m
        
        # 使用形态学操作填充小孔洞
        result = depth_m.copy()
        
        # 创建二值掩码
        mask = (~valid_mask).astype(np.uint8)
        
        # 形态学闭运算填充小孔洞
        kernel_size = self.config.max_hole_size
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        # 使用 inpaint 填充
        # 将深度图转换为适合 inpaint 的格式
        depth_normalized = np.clip(depth_m * 1000, 0, 65535).astype(np.uint16)
        
        # 使用 Telea 算法填充
        try:
            filled = cv2.inpaint(
                depth_normalized.astype(np.float32),
                mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
            result = filled / 1000.0  # 转换回米
        except Exception:
            # 如果 inpaint 失败，使用简单的邻域均值填充
            result = self._simple_fill(depth_m, valid_mask)
        
        return result
    
    def _simple_fill(
        self,
        depth_m: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """简单的邻域均值填充"""
        from scipy.ndimage import uniform_filter
        
        result = depth_m.copy()
        
        # 计算局部均值
        depth_sum = uniform_filter(
            np.where(valid_mask, depth_m, 0),
            size=5
        )
        count = uniform_filter(
            valid_mask.astype(float),
            size=5
        )
        
        # 避免除零
        count[count == 0] = 1
        local_mean = depth_sum / count
        
        # 填充无效区域
        invalid_mask = ~valid_mask
        result[invalid_mask] = local_mean[invalid_mask]
        
        return result
    
    def _edge_preserving_filter(self, depth_m: np.ndarray) -> np.ndarray:
        """
        边缘保持的双边滤波
        """
        # 转换为毫米进行滤波（双边滤波对尺度敏感）
        depth_mm = (depth_m * 1000).astype(np.float32)
        
        # 应用双边滤波
        try:
            filtered = cv2.bilateralFilter(
                depth_mm,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color * 1000,
                sigmaSpace=self.config.bilateral_sigma_space
            )
        except Exception:
            # 如果双边滤波失败，返回原始数据
            return depth_m
        
        # 转换回米
        return filtered / 1000.0
    
    def reset(self):
        """重置处理器状态"""
        self.sampler.reset()
