"""
手部坐标变换器
扩展 CoordinateTransformer 支持手部特有的深度处理
"""
from typing import List, Optional, Tuple

import numpy as np

from .coordinate_transformer import CoordinateTransformer, Point3D
from .pose_estimator import Landmark
from .hand_result import HandResult
from ..hardware.frame_set import Intrinsics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HandCoordinateTransformer(CoordinateTransformer):
    """
    手部关键点坐标变换器
    
    扩展基础变换器，支持手部特有的深度处理：
    - 5x5 中值滤波
    - 深度一致性校验
    - 指尖深度边缘校正
    
    Example:
        >>> transformer = HandCoordinateTransformer(intrinsics)
        >>> points_3d = transformer.hand_landmarks_to_3d(hand_result, depth_frame, image_size)
    """
    
    def __init__(
        self,
        intrinsics: Intrinsics,
        depth_scale: float = 0.001,
        median_filter_size: int = 5,
        consistency_tolerance: float = 0.05,
        depth_sample_quantile: float = 0.5,
        depth_sample_trim_high: float = 0.2
    ):
        """
        初始化手部坐标变换器
        
        Args:
            intrinsics: 相机内参
            depth_scale: 深度值比例因子
            median_filter_size: 中值滤波核大小
            consistency_tolerance: 深度一致性容差 (米)
        """
        super().__init__(
            intrinsics,
            depth_scale,
            median_filter_size=median_filter_size,
            depth_consistency_tolerance=0.15,
            depth_sample_quantile=depth_sample_quantile,
            depth_sample_trim_high=depth_sample_trim_high,
        )
        self.median_filter_size = median_filter_size
        self.consistency_tolerance = consistency_tolerance

    
    def hand_landmarks_to_3d(
        self,
        hand_result: HandResult,
        depth_frame: np.ndarray,
        image_size: Tuple[int, int],
        body_wrist_depth: Optional[float] = None,
        body_wrist_uv: Optional[Tuple[int, int]] = None
    ) -> List[Point3D]:
        """
        将手部2D关键点转换为3D坐标
        
        Args:
            hand_result: 手部检测结果
            depth_frame: 深度图
            image_size: 图像尺寸 (width, height)
            body_wrist_depth: 身体手腕点深度（用于一致性校验）
            body_wrist_uv: 身体手腕像素坐标（用于位置对齐）
        
        Returns:
            21个3D坐标点
        """
        if not hand_result.detected or not hand_result.landmarks:
            return [Point3D(0, 0, 0, confidence=0.0) for _ in range(21)]
        
        width, height = image_size
        points_3d = []
        
        # 首先获取手腕深度作为参考
        wrist_lm = hand_result.landmarks[0]
        wrist_u = int(wrist_lm.x * width)
        wrist_v = int(wrist_lm.y * height)
        hand_wrist_depth = self.apply_median_filter(depth_frame, wrist_u, wrist_v)
        
        # 深度一致性校验 - 优先使用身体手腕深度
        reference_depth = hand_wrist_depth
        use_body_wrist_depth = False
        
        if body_wrist_depth is not None and body_wrist_depth > 0:
            if hand_wrist_depth <= 0:
                # 手部手腕深度无效，使用身体手腕深度
                reference_depth = body_wrist_depth
                use_body_wrist_depth = True
            elif not self.verify_depth_consistency(hand_wrist_depth, body_wrist_depth):
                # 深度不一致，使用身体手腕深度（更可靠）
                logger.debug(f"深度一致性校验失败: 手腕={hand_wrist_depth:.3f}m, 身体={body_wrist_depth:.3f}m，使用身体深度")
                reference_depth = body_wrist_depth
                use_body_wrist_depth = True
        
        # 转换每个关键点
        for i, landmark in enumerate(hand_result.landmarks):
            u = int(landmark.x * width)
            v = int(landmark.y * height)
            
            # 获取深度
            depth = self.apply_median_filter(depth_frame, u, v)
            
            # 指尖深度校正 (索引 4, 8, 12, 16, 20 是指尖)
            if i in [4, 8, 12, 16, 20] and depth > 0:
                depth = self.correct_fingertip_depth(depth_frame, (u, v), reference_depth)
            
            # 如果深度无效，使用参考深度估算
            if depth <= 0 and reference_depth > 0:
                depth = reference_depth
            
            # 手腕点特殊处理：如果使用身体手腕深度，确保一致性
            if i == 0 and use_body_wrist_depth:
                depth = body_wrist_depth
            
            if depth <= 0:
                points_3d.append(Point3D(0, 0, 0, confidence=0.0))
                continue
            
            # 转换为3D坐标
            point = self.pixel_to_3d(u, v, depth)
            point.confidence = landmark.visibility
            points_3d.append(point)
        
        return points_3d
    
    def apply_median_filter(
        self,
        depth_frame: np.ndarray,
        u: int,
        v: int,
        kernel_size: int = None
    ) -> float:
        """
        对指定位置应用中值滤波获取深度
        
        Args:
            depth_frame: 深度图
            u: 像素X坐标
            v: 像素Y坐标
            kernel_size: 滤波核大小，默认使用初始化时的值
        
        Returns:
            滤波后的深度值（米）
        """
        return super().apply_median_filter(depth_frame, u, v, kernel_size=kernel_size)
    
    def verify_depth_consistency(
        self,
        hand_wrist_depth: float,
        body_wrist_depth: float,
        tolerance: float = None
    ) -> bool:
        """
        验证手腕深度一致性
        
        Args:
            hand_wrist_depth: 手部手腕深度
            body_wrist_depth: 身体手腕深度
            tolerance: 容差（米），默认使用初始化时的值
        
        Returns:
            是否一致
        """
        if tolerance is None:
            tolerance = self.consistency_tolerance
        
        if hand_wrist_depth <= 0 or body_wrist_depth <= 0:
            return False
        
        return abs(hand_wrist_depth - body_wrist_depth) <= tolerance
    
    def correct_fingertip_depth(
        self,
        depth_frame: np.ndarray,
        fingertip_uv: Tuple[int, int],
        palm_depth: float
    ) -> float:
        """
        校正指尖深度边缘伪影
        
        指尖处于手指边缘，深度值可能不准确
        使用周围有效深度值进行校正
        
        Args:
            depth_frame: 深度图
            fingertip_uv: 指尖像素坐标 (u, v)
            palm_depth: 手掌参考深度
        
        Returns:
            校正后的深度值
        """
        u, v = fingertip_uv
        height, width = depth_frame.shape[:2]
        
        # 获取指尖处的原始深度
        original_depth = self.apply_median_filter(depth_frame, u, v, kernel_size=3)
        
        if original_depth <= 0:
            return palm_depth
        
        # 检查是否存在边缘伪影（深度突变）
        # 指尖深度不应该比手掌深度远太多或近太多
        max_diff = 0.1  # 10cm
        
        if abs(original_depth - palm_depth) > max_diff:
            # 可能存在边缘伪影，使用更大范围的中值滤波
            corrected_depth = self.apply_median_filter(depth_frame, u, v, kernel_size=7)
            
            if corrected_depth > 0 and abs(corrected_depth - palm_depth) <= max_diff:
                return corrected_depth
            else:
                # 仍然异常，使用手掌深度作为估计
                return palm_depth
        
        return original_depth
    
    def get_body_wrist_depth(
        self,
        body_landmarks: List[Landmark],
        depth_frame: np.ndarray,
        image_size: Tuple[int, int],
        hand: str
    ) -> Optional[float]:
        """
        获取身体手腕点的深度
        
        Args:
            body_landmarks: 身体关键点列表
            depth_frame: 深度图
            image_size: 图像尺寸
            hand: "left" 或 "right"
        
        Returns:
            手腕深度（米），或 None
        """
        # 身体手腕索引: 左手=15, 右手=16
        wrist_idx = 15 if hand == "left" else 16
        
        if wrist_idx >= len(body_landmarks):
            return None
        
        wrist_lm = body_landmarks[wrist_idx]
        if wrist_lm.visibility < 0.5:
            return None
        
        width, height = image_size
        u = int(wrist_lm.x * width)
        v = int(wrist_lm.y * height)
        
        return self.apply_median_filter(depth_frame, u, v)
