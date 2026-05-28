"""
坐标变换器
将2D关键点结合深度信息转换为3D坐标

增强功能：
- 5x5 中值滤波消除深度噪声
- 躯干参考深度一致性校验
- 异常深度值检测与修正
- 解剖学约束验证
"""
from dataclasses import dataclass
import time
from typing import List, Optional, Tuple, Dict

import numpy as np

from ..hardware.frame_set import Intrinsics
from .pose_estimator import Landmark, PoseResult
from ..utils.logger import get_logger

logger = get_logger(__name__)

# 身体关键点分组（用于深度一致性校验）
BODY_REGION_LANDMARKS = {
    'torso': [11, 12, 23, 24],  # 左右肩、左右髋
    'left_arm': [11, 13, 15],   # 左肩、左肘、左腕
    'right_arm': [12, 14, 16],  # 右肩、右肘、右腕
    'left_leg': [23, 25, 27],   # 左髋、左膝、左踝
    'right_leg': [24, 26, 28],  # 右髋、右膝、右踝
    'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 面部关键点
}

# 解剖学骨骼长度比例约束（相对于身高）
BONE_LENGTH_RATIOS = {
    'upper_arm': (0.15, 0.22),    # 上臂长度占身高比例
    'forearm': (0.13, 0.18),      # 前臂长度占身高比例
    'thigh': (0.22, 0.30),        # 大腿长度占身高比例
    'calf': (0.20, 0.28),         # 小腿长度占身高比例
}


@dataclass
class Point3D:
    """3D点"""
    x: float            # X坐标（米），相机坐标系：右为正
    y: float            # Y坐标（米），相机坐标系：下为正
    z: float            # Z坐标（米），相机坐标系：前为正（深度）
    confidence: float = 1.0   # 置信度
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, confidence: float = 1.0) -> 'Point3D':
        """从numpy数组创建"""
        return cls(x=arr[0], y=arr[1], z=arr[2], confidence=confidence)
    
    def distance_to(self, other: 'Point3D') -> float:
        """计算到另一个点的距离"""
        return np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def is_valid(self) -> bool:
        """检查点是否有效"""
        return (
            self.z > 0 and 
            not np.isnan(self.x) and 
            not np.isnan(self.y) and 
            not np.isnan(self.z)
        )


class CoordinateTransformer:
    """
    坐标变换器
    
    将2D图像坐标结合深度信息转换为3D相机坐标
    
    增强功能：
    - 5x5 中值滤波消除深度噪声
    - 躯干参考深度一致性校验
    - 异常深度值检测与修正
    
    Example:
        >>> transformer = CoordinateTransformer(intrinsics)
        >>> point_3d = transformer.pixel_to_3d(320, 240, 1.5)
        >>> points_3d = transformer.landmarks_to_3d_enhanced(pose_result, depth_frame)
    """
    
    def __init__(
        self,
        intrinsics: Intrinsics,
        depth_scale: float = 0.001,
        median_filter_size: int = 5,
        depth_consistency_tolerance: float = 0.15,
        depth_sample_quantile: float = 0.5,
        depth_sample_trim_high: float = 0.2,
        enable_temporal_smoothing: bool = True,
        temporal_smooth_alpha: float = 0.3
    ):
        """
        初始化坐标变换器
        
        Args:
            intrinsics: 相机内参
            depth_scale: 深度值比例因子（将原始深度值转换为米）
            median_filter_size: 中值滤波核大小
            depth_consistency_tolerance: 深度一致性容差（米）
            enable_temporal_smoothing: 是否启用深度时域平滑
            temporal_smooth_alpha: 时域平滑系数（0-1），越小越平滑
        """
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale
        self.median_filter_size = median_filter_size
        self.depth_consistency_tolerance = depth_consistency_tolerance
        self.depth_sample_quantile = depth_sample_quantile
        self.depth_sample_trim_high = depth_sample_trim_high
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.temporal_smooth_alpha = temporal_smooth_alpha
        self._last_torso_reference_warning_ts = 0.0
        self._torso_reference_warning_interval_sec = 2.0
        
        # 深度时域平滑状态：缓存每个关键点的上一帧深度值
        self._prev_depths: Dict[int, float] = {}
        
        # 预计算
        self._fx_inv = 1.0 / intrinsics.fx
        self._fy_inv = 1.0 / intrinsics.fy
        
        logger.debug(f"CoordinateTransformer 初始化: {intrinsics.width}x{intrinsics.height}, "
                    f"median_filter={median_filter_size}")
    
    def pixel_to_3d(self, u: int, v: int, depth: float) -> Point3D:
        """
        像素坐标转3D坐标
        
        Args:
            u: 像素X坐标
            v: 像素Y坐标
            depth: 深度值（米）
        
        Returns:
            Point3D: 3D坐标
        """
        if depth <= 0:
            return Point3D(0, 0, 0, confidence=0.0)
        
        x = (u - self.intrinsics.ppx) * depth * self._fx_inv
        y = (v - self.intrinsics.ppy) * depth * self._fy_inv
        z = depth
        
        return Point3D(x, y, z, confidence=1.0)
    
    def get_depth_at_pixel(
        self,
        depth_frame: np.ndarray,
        u: int,
        v: int,
        search_radius: int = 3
    ) -> float:
        """
        获取像素位置的深度值（旧版，保留兼容性）
        
        如果该位置深度无效，会在邻域内搜索有效深度
        
        Args:
            depth_frame: 深度图
            u: 像素X坐标
            v: 像素Y坐标
            search_radius: 搜索半径
        
        Returns:
            深度值（米），无效返回0
        """
        height, width = depth_frame.shape[:2]
        
        # 边界检查
        if not (0 <= u < width and 0 <= v < height):
            return 0.0
        
        # 直接获取
        depth_raw = depth_frame[v, u]
        
        # 转换为米
        if depth_frame.dtype == np.uint16:
            depth = float(depth_raw) * self.depth_scale
        else:
            depth = float(depth_raw)
        
        # 如果有效，直接返回
        if depth > 0:
            return depth
        
        # 在邻域内搜索有效深度
        if search_radius > 0:
            for r in range(1, search_radius + 1):
                depths = []
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny, nx = v + dy, u + dx
                        if 0 <= nx < width and 0 <= ny < height:
                            d_raw = depth_frame[ny, nx]
                            if depth_frame.dtype == np.uint16:
                                d = float(d_raw) * self.depth_scale
                            else:
                                d = float(d_raw)
                            if d > 0:
                                depths.append(d)
                
                if depths:
                    # 返回邻域中值
                    return float(np.median(depths))
        
        return 0.0
    
    def apply_median_filter(
        self,
        depth_frame: np.ndarray,
        u: int,
        v: int,
        kernel_size: int = None
    ) -> float:
        """
        对指定位置应用中值滤波获取深度（增强版）
        
        Args:
            depth_frame: 深度图
            u: 像素X坐标
            v: 像素Y坐标
            kernel_size: 滤波核大小，默认使用初始化时的值
        
        Returns:
            滤波后的深度值（米）
        """
        if kernel_size is None:
            kernel_size = self.median_filter_size
        
        height, width = depth_frame.shape[:2]
        half_k = kernel_size // 2
        
        # 边界检查
        if not (0 <= u < width and 0 <= v < height):
            return 0.0
        
        # 收集邻域深度值
        depths = []
        for dy in range(-half_k, half_k + 1):
            for dx in range(-half_k, half_k + 1):
                ny, nx = v + dy, u + dx
                if 0 <= nx < width and 0 <= ny < height:
                    d_raw = depth_frame[ny, nx]
                    if depth_frame.dtype == np.uint16:
                        d = float(d_raw) * self.depth_scale
                    else:
                        d = float(d_raw)
                    if d > 0:
                        depths.append(d)
        
        if not depths:
            return 0.0

        depths_arr = np.asarray(depths, dtype=np.float32)

        if self.depth_sample_trim_high > 0 and depths_arr.size >= 8:
            trim_high_q = 1.0 - float(self.depth_sample_trim_high)
            trim_high_q = max(0.05, min(trim_high_q, 1.0))
            high_cut = float(np.quantile(depths_arr, trim_high_q))
            depths_arr = depths_arr[depths_arr <= high_cut]
            if depths_arr.size == 0:
                return 0.0

        q = self.depth_sample_quantile
        if q is None:
            return float(np.median(depths_arr))
        q = float(q)
        q = max(0.0, min(q, 1.0))
        return float(np.quantile(depths_arr, q))
    
    def get_torso_reference_depth(
        self,
        pose_result: PoseResult,
        depth_frame: np.ndarray,
        image_size: Tuple[int, int]
    ) -> float:
        """
        获取躯干参考深度
        
        使用左右肩和左右髋的中值深度作为参考
        
        Args:
            pose_result: 姿态检测结果
            depth_frame: 深度图
            image_size: 图像尺寸 (width, height)
        
        Returns:
            躯干参考深度（米）
        """
        width, height = image_size
        torso_indices = BODY_REGION_LANDMARKS['torso']  # [11, 12, 23, 24]
        
        depths = []
        for idx in torso_indices:
            if idx < len(pose_result.landmarks):
                lm = pose_result.landmarks[idx]
                if lm.visibility >= 0.5:
                    u = int(lm.x * width)
                    v = int(lm.y * height)
                    d = self.apply_median_filter(depth_frame, u, v)
                    if d > 0:
                        depths.append(d)
        
        if not depths:
            return 0.0
        
        return float(np.median(depths))
    
    def verify_depth_consistency(
        self,
        depth: float,
        reference_depth: float,
        tolerance: float = None
    ) -> bool:
        """
        验证深度一致性
        
        Args:
            depth: 待验证的深度值
            reference_depth: 参考深度值
            tolerance: 容差（米），默认使用初始化时的值
        
        Returns:
            是否一致
        """
        if tolerance is None:
            tolerance = self.depth_consistency_tolerance
        
        if depth <= 0 or reference_depth <= 0:
            return False
        
        return abs(depth - reference_depth) <= tolerance
    
    def correct_anomalous_depth(
        self,
        depth: float,
        reference_depth: float,
        landmark_idx: int
    ) -> Tuple[float, bool]:
        """
        校正异常深度值
        
        Args:
            depth: 原始深度值
            reference_depth: 参考深度值
            landmark_idx: 关键点索引
        
        Returns:
            (校正后的深度, 是否进行了校正)
        """
        if depth <= 0:
            return reference_depth, True
        
        if reference_depth <= 0:
            return depth, False
        
        # 检查深度是否异常（偏差超过容差）
        if not self.verify_depth_consistency(depth, reference_depth):
            # 根据关键点类型决定校正策略
            # 四肢末端允许更大的深度变化
            limb_end_indices = [
                13, 14,            # elbows
                15, 16,            # wrists
                19, 20, 21, 22,    # hand points
                25, 26,            # knees
                27, 28, 29, 30, 31, 32  # ankles + feet
            ]
            
            if landmark_idx in limb_end_indices:
                # 四肢末端：允许更大偏差，但仍需限制
                max_deviation = 0.5  # 50cm
                if abs(depth - reference_depth) > max_deviation:
                    logger.debug(f"关键点 {landmark_idx} 深度异常: {depth:.3f}m -> {reference_depth:.3f}m")
                    return reference_depth, True
            else:
                # 躯干和四肢近端：使用参考深度
                logger.debug(f"关键点 {landmark_idx} 深度异常: {depth:.3f}m -> {reference_depth:.3f}m")
                return reference_depth, True
        
        return depth, False
    
    def landmark_to_3d(
        self,
        landmark: Landmark,
        depth_frame: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Point3D:
        """
        将单个2D关键点转换为3D坐标
        
        Args:
            landmark: 2D关键点
            depth_frame: 深度图
            image_size: 图像尺寸 (width, height)
        
        Returns:
            Point3D: 3D坐标
        """
        width, height = image_size
        
        # 归一化坐标转像素坐标
        u = int(landmark.x * width)
        v = int(landmark.y * height)
        
        # 获取深度
        depth = self.get_depth_at_pixel(depth_frame, u, v)
        
        if depth <= 0:
            # 深度无效，返回低置信度点
            return Point3D(0, 0, 0, confidence=0.0)
        
        # 转换为3D
        point = self.pixel_to_3d(u, v, depth)
        point.confidence = landmark.visibility
        
        return point
    
    def landmarks_to_3d(
        self,
        pose_result: PoseResult,
        depth_frame: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None
    ) -> List[Point3D]:
        """
        将所有2D关键点转换为3D坐标（旧版，保留兼容性）
        
        Args:
            pose_result: 姿态检测结果
            depth_frame: 深度图
            image_size: 图像尺寸 (width, height)，默认使用内参尺寸
        
        Returns:
            List[Point3D]: 33个3D坐标点
        """
        if image_size is None:
            image_size = (self.intrinsics.width, self.intrinsics.height)
        
        points_3d = []
        
        for landmark in pose_result.landmarks:
            point = self.landmark_to_3d(landmark, depth_frame, image_size)
            points_3d.append(point)
        
        return points_3d
    
    def landmarks_to_3d_enhanced(
        self,
        pose_result: PoseResult,
        depth_frame: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
        min_visibility: float = 0.5,
        depth_range: Tuple[float, float] = (0.3, 4.0)
    ) -> List[Point3D]:
        """
        增强版：将所有2D关键点转换为3D坐标
        
        增强功能：
        1. 5x5 中值滤波消除深度噪声
        2. 躯干参考深度一致性校验
        3. 异常深度值检测与修正
        
        Args:
            pose_result: 姿态检测结果
            depth_frame: 深度图
            image_size: 图像尺寸 (width, height)
            min_visibility: 最小可见性阈值
            depth_range: 有效深度范围 (min, max)
        
        Returns:
            List[Point3D]: 33个3D坐标点
        """
        if image_size is None:
            image_size = (self.intrinsics.width, self.intrinsics.height)
        
        width, height = image_size
        min_depth, max_depth = depth_range
        
        # 1. 获取躯干参考深度
        reference_depth = self.get_torso_reference_depth(pose_result, depth_frame, image_size)
        if reference_depth <= 0:
            now = time.monotonic()
            if (now - self._last_torso_reference_warning_ts) >= self._torso_reference_warning_interval_sec:
                logger.warning("无法获取躯干参考深度，使用基础转换")
                self._last_torso_reference_warning_ts = now
            return self.landmarks_to_3d(pose_result, depth_frame, image_size)
        
        logger.debug(f"躯干参考深度: {reference_depth:.3f}m")
        
        # 2. 转换每个关键点
        points_3d = []
        corrections_made = 0
        
        for idx, landmark in enumerate(pose_result.landmarks):
            # 可见性检查
            if landmark.visibility < min_visibility:
                points_3d.append(Point3D(0, 0, 0, confidence=0.0))
                continue
            
            # 像素坐标
            u = int(landmark.x * width)
            v = int(landmark.y * height)
            
            # 使用中值滤波获取深度
            depth = self.apply_median_filter(depth_frame, u, v)
            
            # 深度范围检查
            if depth < min_depth or depth > max_depth:
                depth = 0.0
            
            # 深度一致性校验和校正
            if depth > 0:
                corrected_depth, was_corrected = self.correct_anomalous_depth(
                    depth, reference_depth, idx
                )
                if was_corrected:
                    corrections_made += 1
                depth = corrected_depth
            else:
                # 深度无效，使用参考深度
                depth = reference_depth
                corrections_made += 1
            
            # 深度时域平滑：与上一帧深度值进行指数加权平均
            if self.enable_temporal_smoothing and depth > 0:
                prev_depth = self._prev_depths.get(idx)
                if prev_depth is not None and prev_depth > 0:
                    # 只在深度变化合理时平滑（防止人体移动时过度平滑）
                    depth_change = abs(depth - prev_depth)
                    if depth_change < 0.1:  # 变化小于10cm时平滑
                        alpha = self.temporal_smooth_alpha
                        depth = alpha * depth + (1 - alpha) * prev_depth
                self._prev_depths[idx] = depth
            
            # 转换为3D坐标
            if depth > 0:
                point = self.pixel_to_3d(u, v, depth)
                point.confidence = landmark.visibility
            else:
                point = Point3D(0, 0, 0, confidence=0.0)
            
            points_3d.append(point)
        
        if corrections_made > 0:
            logger.debug(f"深度校正: {corrections_made} 个关键点")
        
        return points_3d
    
    def transform_with_filter(
        self,
        pose_result: PoseResult,
        depth_frame: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
        min_visibility: float = 0.5,
        depth_range: Tuple[float, float] = (0.3, 3.0),
        use_enhanced: bool = True
    ) -> List[Point3D]:
        """
        带过滤的坐标变换（默认使用增强版）
        
        增强功能：
        1. 5x5 中值滤波消除深度噪声
        2. 躯干参考深度一致性校验
        3. 异常深度值检测与修正
        
        Args:
            pose_result: 姿态检测结果
            depth_frame: 深度图
            image_size: 图像尺寸
            min_visibility: 最小可见性阈值
            depth_range: 有效深度范围 (min, max)
            use_enhanced: 是否使用增强版深度处理
        
        Returns:
            List[Point3D]: 3D坐标点列表
        """
        if image_size is None:
            image_size = (self.intrinsics.width, self.intrinsics.height)
        
        # 使用增强版转换
        if use_enhanced:
            return self.landmarks_to_3d_enhanced(
                pose_result, depth_frame, image_size,
                min_visibility=min_visibility,
                depth_range=depth_range
            )
        
        # 旧版逻辑（保留兼容性）
        points_3d = []
        min_depth, max_depth = depth_range
        
        for landmark in pose_result.landmarks:
            # 可见性检查
            if landmark.visibility < min_visibility:
                points_3d.append(Point3D(0, 0, 0, confidence=0.0))
                continue
            
            # 转换
            point = self.landmark_to_3d(landmark, depth_frame, image_size)
            
            # 深度范围检查
            if point.z < min_depth or point.z > max_depth:
                point.confidence *= 0.5  # 降低置信度
            
            points_3d.append(point)
        
        return points_3d


def calculate_angle(p1: Point3D, p2: Point3D, p3: Point3D) -> float:
    """
    计算三点形成的角度（p2为顶点）
    
    Args:
        p1, p2, p3: 三个3D点，p2是角的顶点
    
    Returns:
        角度（度）
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    # 计算向量模
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 计算夹角
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_distance(p1: Point3D, p2: Point3D) -> float:
    """
    计算两点之间的欧几里得距离
    
    Args:
        p1, p2: 两个3D点
    
    Returns:
        距离（米）
    """
    return p1.distance_to(p2)
