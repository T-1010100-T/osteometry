"""
帧数据结构定义
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Intrinsics:
    """
    相机内参
    
    用于像素坐标和3D坐标之间的转换
    """
    width: int              # 图像宽度
    height: int             # 图像高度
    fx: float               # X方向焦距
    fy: float               # Y方向焦距
    ppx: float              # 主点X坐标 (光心)
    ppy: float              # 主点Y坐标 (光心)
    coeffs: List[float] = field(default_factory=lambda: [0.0] * 5)  # 畸变系数
    
    def pixel_to_point(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """
        像素坐标转3D点（相机坐标系）
        
        Args:
            u: 像素X坐标
            v: 像素Y坐标
            depth: 深度值（米）
        
        Returns:
            (x, y, z): 3D坐标（米）
        """
        if depth <= 0:
            return (0.0, 0.0, 0.0)
        
        x = (u - self.ppx) * depth / self.fx
        y = (v - self.ppy) * depth / self.fy
        z = depth
        
        return (x, y, z)
    
    def point_to_pixel(self, x: float, y: float, z: float) -> Tuple[int, int]:
        """
        3D点转像素坐标
        
        Args:
            x, y, z: 3D坐标（米）
        
        Returns:
            (u, v): 像素坐标
        """
        if z <= 0:
            return (0, 0)
        
        u = int(x * self.fx / z + self.ppx)
        v = int(y * self.fy / z + self.ppy)
        
        return (u, v)
    
    def to_matrix(self) -> np.ndarray:
        """
        转换为相机内参矩阵 (3x3)
        
        Returns:
            内参矩阵 K
        """
        return np.array([
            [self.fx, 0, self.ppx],
            [0, self.fy, self.ppy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass
class FrameSet:
    """
    RGB-D 帧数据
    
    封装单帧的 RGB 图像和深度图
    """
    color_frame: np.ndarray     # RGB图像, shape: (H, W, 3), dtype: uint8
    depth_frame: np.ndarray     # 深度图, shape: (H, W), dtype: uint16 或 float32
    timestamp: float            # 时间戳（秒）
    frame_number: int           # 帧序号
    intrinsics: Optional[Intrinsics] = None  # 相机内参
    
    @property
    def width(self) -> int:
        """图像宽度"""
        return self.color_frame.shape[1]
    
    @property
    def height(self) -> int:
        """图像高度"""
        return self.color_frame.shape[0]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """图像尺寸 (height, width)"""
        return (self.height, self.width)
    
    def get_depth_at(self, u: int, v: int) -> float:
        """
        获取指定像素的深度值
        
        Args:
            u: 像素X坐标
            v: 像素Y坐标
        
        Returns:
            深度值（米），无效返回0
        """
        if 0 <= u < self.width and 0 <= v < self.height:
            depth = self.depth_frame[v, u]
            # 如果深度是 uint16 格式（毫米），转换为米
            if self.depth_frame.dtype == np.uint16:
                return float(depth) * 0.001
            return float(depth)
        return 0.0
    
    def get_point_3d(self, u: int, v: int) -> Optional[Tuple[float, float, float]]:
        """
        获取指定像素的3D坐标
        
        Args:
            u: 像素X坐标
            v: 像素Y坐标
        
        Returns:
            (x, y, z) 3D坐标，或 None（如果无效）
        """
        if self.intrinsics is None:
            return None
        
        depth = self.get_depth_at(u, v)
        if depth <= 0:
            return None
        
        return self.intrinsics.pixel_to_point(u, v, depth)
    
    def is_valid(self) -> bool:
        """检查帧数据是否有效"""
        return (
            self.color_frame is not None and
            self.depth_frame is not None and
            self.color_frame.size > 0 and
            self.depth_frame.size > 0
        )
