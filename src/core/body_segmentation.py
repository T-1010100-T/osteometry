"""
人体深度分割模块

基于MediaPipe关键点生成人体掩码，只采样人体区域的深度值，排除背景干扰。

原理：
1. 使用MediaPipe检测到的33个身体关键点
2. 构建人体轮廓多边形（凸包或骨骼连接区域）
3. 生成二值掩码
4. 深度采样时只考虑掩码内的像素
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .pose_estimator import PoseResult, Landmark
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentationConfig:
    """分割配置"""
    # 轮廓扩展像素（向外扩展以包含衣服厚度）
    contour_expand_pixels: int = 15
    # 最小可见性阈值
    min_visibility: float = 0.3
    # 是否使用凸包（True=凸包，False=骨骼区域）
    use_convex_hull: bool = False
    # 深度容差（米）- 只保留与参考深度接近的像素
    depth_tolerance: float = 0.3
    # 是否启用深度过滤
    enable_depth_filter: bool = True


# 人体轮廓关键点索引（按顺序连接形成轮廓）
# 从头顶开始，顺时针绕一圈
BODY_CONTOUR_INDICES = [
    # 头部
    0,   # 鼻子（头顶参考）
    # 右侧
    8,   # 右耳
    12,  # 右肩
    14,  # 右肘
    16,  # 右腕
    20,  # 右手食指
    22,  # 右手拇指
    16,  # 右腕（返回）
    14,  # 右肘（返回）
    12,  # 右肩（返回）
    24,  # 右髋
    26,  # 右膝
    28,  # 右踝
    32,  # 右脚
    30,  # 右脚跟
    # 左侧（反向）
    29,  # 左脚跟
    31,  # 左脚
    27,  # 左踝
    25,  # 左膝
    23,  # 左髋
    11,  # 左肩
    13,  # 左肘
    15,  # 左腕
    19,  # 左手食指
    21,  # 左手拇指
    15,  # 左腕（返回）
    13,  # 左肘（返回）
    11,  # 左肩（返回）
    7,   # 左耳
]

# 简化版轮廓（只用主要关键点）
SIMPLE_CONTOUR_INDICES = [
    0,   # 头
    12,  # 右肩
    24,  # 右髋
    26,  # 右膝
    28,  # 右踝
    27,  # 左踝
    25,  # 左膝
    23,  # 左髋
    11,  # 左肩
]


class BodySegmentation:
    """
    人体深度分割器
    
    功能：
    1. 根据MediaPipe关键点生成人体掩码
    2. 过滤背景深度，只保留人体区域
    3. 支持深度范围过滤（排除前景/背景干扰物）
    """
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        self._last_mask: Optional[np.ndarray] = None
        self._last_reference_depth: float = 0.0
    
    def create_body_mask(
        self,
        pose_result: PoseResult,
        image_size: Tuple[int, int],
        reference_depth: Optional[float] = None,
        depth_frame: Optional[np.ndarray] = None,
        depth_scale: float = 0.001
    ) -> np.ndarray:
        """
        创建人体掩码
        
        Args:
            pose_result: MediaPipe姿态检测结果
            image_size: 图像尺寸 (width, height)
            reference_depth: 参考深度（米），用于深度过滤
            depth_frame: 深度图（可选，用于深度过滤）
            depth_scale: 深度比例因子
        
        Returns:
            二值掩码 (H, W)，人体区域为255，背景为0
        """
        width, height = image_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if not pose_result.detected:
            return mask
        
        # 收集有效的轮廓点
        contour_points = []
        indices = SIMPLE_CONTOUR_INDICES if not self.config.use_convex_hull else range(33)
        
        for idx in indices:
            if idx < len(pose_result.landmarks):
                lm = pose_result.landmarks[idx]
                if lm.visibility >= self.config.min_visibility:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    # 边界检查
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    contour_points.append([x, y])
        
        if len(contour_points) < 3:
            # 点太少，无法形成轮廓
            return mask
        
        contour_points = np.array(contour_points, dtype=np.int32)
        
        # 生成轮廓
        if self.config.use_convex_hull:
            # 使用凸包
            hull = cv2.convexHull(contour_points)
            contour = hull
        else:
            # 使用原始轮廓
            contour = contour_points.reshape((-1, 1, 2))
        
        # 填充轮廓
        cv2.fillPoly(mask, [contour], 255)
        
        # 扩展轮廓（包含衣服厚度）
        if self.config.contour_expand_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.contour_expand_pixels * 2 + 1,
                 self.config.contour_expand_pixels * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel)
        
        # 深度过滤（可选）
        if (self.config.enable_depth_filter and 
            reference_depth is not None and 
            reference_depth > 0 and
            depth_frame is not None):
            mask = self._apply_depth_filter(
                mask, depth_frame, reference_depth, depth_scale
            )
        
        self._last_mask = mask
        self._last_reference_depth = reference_depth or 0.0
        
        return mask
    
    def _apply_depth_filter(
        self,
        mask: np.ndarray,
        depth_frame: np.ndarray,
        reference_depth: float,
        depth_scale: float
    ) -> np.ndarray:
        """
        应用深度过滤，排除深度异常的像素
        
        只保留深度在 [reference - tolerance, reference + tolerance] 范围内的像素
        """
        # 转换深度图为米
        if depth_frame.dtype == np.uint16:
            depth_m = depth_frame.astype(np.float32) * depth_scale
        else:
            depth_m = depth_frame.astype(np.float32)
        
        # 计算深度范围
        min_depth = reference_depth - self.config.depth_tolerance
        max_depth = reference_depth + self.config.depth_tolerance
        
        # 创建深度有效掩码
        depth_valid = (depth_m >= min_depth) & (depth_m <= max_depth) & (depth_m > 0)
        
        # 与人体掩码取交集
        result = mask.copy()
        result[~depth_valid] = 0
        
        return result
    
    def get_masked_depth(
        self,
        depth_frame: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        获取掩码区域的深度图
        
        Args:
            depth_frame: 原始深度图
            mask: 人体掩码
        
        Returns:
            掩码后的深度图（背景区域设为0）
        """
        result = depth_frame.copy()
        result[mask == 0] = 0
        return result
    
    def sample_depth_in_region(
        self,
        depth_frame: np.ndarray,
        mask: np.ndarray,
        center_x: int,
        center_y: int,
        window_size: int = 5,
        depth_scale: float = 0.001
    ) -> float:
        """
        在掩码区域内采样深度
        
        只采样掩码内的有效深度值
        
        Args:
            depth_frame: 深度图
            mask: 人体掩码
            center_x, center_y: 采样中心点
            window_size: 采样窗口大小
            depth_scale: 深度比例因子
        
        Returns:
            深度值（米），无效返回0
        """
        height, width = depth_frame.shape[:2]
        half_w = window_size // 2
        
        # 收集窗口内掩码有效区域的深度值
        depths = []
        for dy in range(-half_w, half_w + 1):
            for dx in range(-half_w, half_w + 1):
                y, x = center_y + dy, center_x + dx
                if 0 <= x < width and 0 <= y < height:
                    # 检查是否在掩码内
                    if mask[y, x] > 0:
                        d_raw = depth_frame[y, x]
                        if depth_frame.dtype == np.uint16:
                            d = float(d_raw) * depth_scale
                        else:
                            d = float(d_raw)
                        if d > 0:
                            depths.append(d)
        
        if not depths:
            return 0.0
        
        # 返回中值
        return float(np.median(depths))
    
    def get_body_depth_stats(
        self,
        depth_frame: np.ndarray,
        mask: np.ndarray,
        depth_scale: float = 0.001
    ) -> Dict[str, float]:
        """
        获取人体区域的深度统计信息
        
        Returns:
            包含 min, max, mean, median, std 的字典
        """
        # 获取掩码区域的深度值
        if depth_frame.dtype == np.uint16:
            depth_m = depth_frame.astype(np.float32) * depth_scale
        else:
            depth_m = depth_frame.astype(np.float32)
        
        # 只取掩码内的有效深度
        valid_depths = depth_m[(mask > 0) & (depth_m > 0)]
        
        if len(valid_depths) == 0:
            return {
                'min': 0.0, 'max': 0.0, 'mean': 0.0,
                'median': 0.0, 'std': 0.0, 'count': 0
            }
        
        return {
            'min': float(np.min(valid_depths)),
            'max': float(np.max(valid_depths)),
            'mean': float(np.mean(valid_depths)),
            'median': float(np.median(valid_depths)),
            'std': float(np.std(valid_depths)),
            'count': len(valid_depths)
        }
    
    @property
    def last_mask(self) -> Optional[np.ndarray]:
        """获取最近一次生成的掩码"""
        return self._last_mask
    
    def reset(self):
        """重置状态"""
        self._last_mask = None
        self._last_reference_depth = 0.0
