"""
手部检测ROI优化器

基于身体手腕位置创建手部检测区域，提高检测效率

**Feature: holistic-hand-integration**
**Property 14: ROI计算正确性**
**Validates: Requirements 8.3, 8.4**
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .pose_estimator import PoseResult, Landmark
from .constants import PoseLandmark
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ROI:
    """ROI区域"""
    x: int      # 左上角X
    y: int      # 左上角Y
    width: int  # 宽度
    height: int # 高度
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """转换为元组 (x, y, width, height)"""
        return (self.x, self.y, self.width, self.height)
    
    def contains(self, px: int, py: int) -> bool:
        """检查点是否在ROI内"""
        return (self.x <= px < self.x + self.width and
                self.y <= py < self.y + self.height)


class ROIOptimizer:
    """
    手部检测ROI优化器
    
    基于身体手腕位置创建手部检测区域，减少全帧检测开销
    
    **Property 14: ROI计算正确性**
    
    Example:
        >>> optimizer = ROIOptimizer(roi_size=300)
        >>> roi = optimizer.get_hand_roi(pose_result, (640, 480), "left")
        >>> if roi:
        ...     cropped = optimizer.crop_roi(image, roi)
    """
    
    # 默认ROI尺寸
    DEFAULT_ROI_SIZE = 300
    
    # ROI扩展系数（相对于手腕位置）
    ROI_OFFSET_FACTOR = 0.3  # ROI中心相对手腕向手指方向偏移
    
    def __init__(self, roi_size: int = DEFAULT_ROI_SIZE):
        """
        初始化ROI优化器
        
        Args:
            roi_size: ROI尺寸（像素）
        """
        self.roi_size = roi_size
        
        # 上一帧的ROI（用于平滑）
        self._last_left_roi: Optional[ROI] = None
        self._last_right_roi: Optional[ROI] = None
    
    def get_hand_roi(
        self,
        pose_result: PoseResult,
        image_size: Tuple[int, int],
        hand: str  # "left" / "right"
    ) -> Optional[ROI]:
        """
        获取手部ROI区域
        
        **Property 14: ROI计算正确性**
        以手腕为中心，创建roi_size x roi_size的区域，裁剪到图像边界
        
        Args:
            pose_result: 身体检测结果
            image_size: 图像尺寸 (width, height)
            hand: 左手或右手
        
        Returns:
            ROI区域，或 None（如果手腕不可见）
        """
        if not pose_result.detected:
            return None
        
        width, height = image_size
        
        # 获取手腕关键点
        wrist_idx = PoseLandmark.LEFT_WRIST if hand == "left" else PoseLandmark.RIGHT_WRIST
        
        if wrist_idx >= len(pose_result.landmarks):
            return None
        
        wrist = pose_result.landmarks[wrist_idx]
        
        # 检查可见性
        if wrist.visibility < 0.5:
            # 使用上一帧的ROI
            last_roi = self._last_left_roi if hand == "left" else self._last_right_roi
            return last_roi
        
        # 计算像素坐标
        wrist_x = int(wrist.x * width)
        wrist_y = int(wrist.y * height)
        
        # 获取肘部位置用于确定手的方向
        elbow_idx = PoseLandmark.LEFT_ELBOW if hand == "left" else PoseLandmark.RIGHT_ELBOW
        offset_x, offset_y = 0, 0
        
        if elbow_idx < len(pose_result.landmarks):
            elbow = pose_result.landmarks[elbow_idx]
            if elbow.visibility > 0.5:
                # 手的方向是从肘到手腕的延伸
                elbow_x = int(elbow.x * width)
                elbow_y = int(elbow.y * height)
                
                dx = wrist_x - elbow_x
                dy = wrist_y - elbow_y
                
                # 向手指方向偏移ROI中心
                offset_x = int(dx * self.ROI_OFFSET_FACTOR)
                offset_y = int(dy * self.ROI_OFFSET_FACTOR)
        
        # 计算ROI中心
        center_x = wrist_x + offset_x
        center_y = wrist_y + offset_y
        
        # 计算ROI边界
        half_size = self.roi_size // 2
        roi_x = center_x - half_size
        roi_y = center_y - half_size
        
        # 裁剪到图像边界
        roi_x = max(0, min(roi_x, width - self.roi_size))
        roi_y = max(0, min(roi_y, height - self.roi_size))
        
        # 确保ROI不超出图像
        roi_width = min(self.roi_size, width - roi_x)
        roi_height = min(self.roi_size, height - roi_y)
        
        roi = ROI(x=roi_x, y=roi_y, width=roi_width, height=roi_height)
        
        # 保存ROI
        if hand == "left":
            self._last_left_roi = roi
        else:
            self._last_right_roi = roi
        
        return roi
    
    def crop_roi(
        self,
        image: np.ndarray,
        roi: ROI
    ) -> np.ndarray:
        """
        裁剪ROI区域
        
        Args:
            image: 输入图像
            roi: ROI区域
        
        Returns:
            裁剪后的图像
        """
        return image[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width].copy()
    
    def transform_landmarks_to_original(
        self,
        landmarks: List[Landmark],
        roi: ROI,
        image_size: Tuple[int, int]
    ) -> List[Landmark]:
        """
        将ROI内的关键点坐标转换回原图坐标
        
        Args:
            landmarks: ROI内的关键点（归一化坐标）
            roi: ROI区域
            image_size: 原图尺寸 (width, height)
        
        Returns:
            转换后的关键点列表
        """
        width, height = image_size
        transformed = []
        
        for lm in landmarks:
            # ROI内的像素坐标
            roi_px = lm.x * roi.width
            roi_py = lm.y * roi.height
            
            # 原图像素坐标
            orig_px = roi.x + roi_px
            orig_py = roi.y + roi_py
            
            # 归一化到原图
            new_x = orig_px / width
            new_y = orig_py / height
            
            transformed.append(Landmark(
                x=new_x,
                y=new_y,
                z=lm.z,
                visibility=lm.visibility
            ))
        
        return transformed
    
    def reset(self) -> None:
        """重置状态"""
        self._last_left_roi = None
        self._last_right_roi = None
    
    @property
    def last_left_roi(self) -> Optional[ROI]:
        """上一帧左手ROI"""
        return self._last_left_roi
    
    @property
    def last_right_roi(self) -> Optional[ROI]:
        """上一帧右手ROI"""
        return self._last_right_roi
