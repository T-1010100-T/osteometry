"""
关键点稳定器模块

使用 One-Euro 滤波器稳定 2D 和 3D 关键点
"""
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from src.core.stabilizer_config import StabilizerConfig, OneEuroConfig

if TYPE_CHECKING:
    from src.core.coordinate_transformer import Point3D
    from src.core.pose_estimator import Landmark


class LowPassFilter:
    """
    简单低通滤波器
    
    y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        初始化
        
        Args:
            alpha: 平滑系数 (0-1)，值越大响应越快
        """
        self.alpha = alpha
        self.y_prev: Optional[float] = None
        self.initialized = False
    
    def __call__(self, value: float, alpha: Optional[float] = None) -> float:
        """
        滤波
        
        Args:
            value: 输入值
            alpha: 可选的临时平滑系数
        
        Returns:
            滤波后的值
        """
        a = alpha if alpha is not None else self.alpha
        
        if not self.initialized:
            self.y_prev = value
            self.initialized = True
            return value
        
        y = a * value + (1 - a) * self.y_prev
        self.y_prev = y
        return y
    
    def reset(self):
        """重置滤波器"""
        self.y_prev = None
        self.initialized = False
    
    @property
    def last_value(self) -> Optional[float]:
        """获取上一个滤波值"""
        return self.y_prev


class OneEuroFilter:
    """
    One-Euro 滤波器
    
    自适应低通滤波器，根据信号变化速度调整截止频率：
    - 静止时：低截止频率，强平滑
    - 运动时：高截止频率，弱平滑（减少延迟）
    
    参考：https://hal.inria.fr/hal-00670496/document
    """
    
    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 0.05,
        dcutoff: float = 1.0
    ):
        """
        初始化
        
        Args:
            freq: 采样频率（Hz）
            mincutoff: 最小截止频率（Hz），控制静止时的平滑强度
            beta: 速度系数，控制运动时截止频率的增加速度
            dcutoff: 导数截止频率（Hz），用于平滑速度估计
        """
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        
        # 值滤波器
        self.x_filter = LowPassFilter()
        # 导数滤波器
        self.dx_filter = LowPassFilter()
        
        self.last_time: Optional[float] = None
        self.initialized = False
    
    @staticmethod
    def _alpha(cutoff: float, te: float) -> float:
        """
        计算平滑系数
        
        Args:
            cutoff: 截止频率（Hz）
            te: 采样周期（秒）
        
        Returns:
            平滑系数 alpha
        """
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def __call__(self, value: float, timestamp: Optional[float] = None) -> float:
        """
        滤波
        
        Args:
            value: 输入值
            timestamp: 时间戳（秒），如果为 None 则使用默认频率
        
        Returns:
            滤波后的值
        """
        # 计算采样周期
        if timestamp is not None and self.last_time is not None:
            te = timestamp - self.last_time
            if te <= 0:
                te = 1.0 / self.freq
        else:
            te = 1.0 / self.freq
        
        if timestamp is not None:
            self.last_time = timestamp
        
        # 首次调用
        if not self.initialized:
            self.x_filter.y_prev = value
            self.x_filter.initialized = True
            self.dx_filter.y_prev = 0.0
            self.dx_filter.initialized = True
            self.initialized = True
            return value
        
        # 计算导数（速度）
        dx = (value - self.x_filter.y_prev) / te
        
        # 滤波导数
        alpha_d = self._alpha(self.dcutoff, te)
        dx_filtered = self.dx_filter(dx, alpha_d)
        
        # 自适应截止频率
        cutoff = self.mincutoff + self.beta * abs(dx_filtered)
        
        # 滤波值
        alpha = self._alpha(cutoff, te)
        return self.x_filter(value, alpha)
    
    def reset(self):
        """重置滤波器"""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None
        self.initialized = False
    
    @property
    def last_value(self) -> Optional[float]:
        """获取上一个滤波值"""
        return self.x_filter.last_value


@dataclass
class KeypointHistory:
    """关键点历史记录"""
    values_2d: deque  # (x, y, confidence, timestamp)
    values_3d: deque  # (x, y, z, confidence, timestamp)
    
    def __init__(self, max_size: int = 5):
        self.values_2d = deque(maxlen=max_size)
        self.values_3d = deque(maxlen=max_size)


class KeypointStabilizer:
    """
    关键点稳定器
    
    使用 One-Euro 滤波器稳定 2D 和 3D 关键点
    
    功能：
    - 2D 关键点稳定（像素坐标）
    - 3D 关键点稳定（米坐标）
    - 置信度自适应调整
    - 异常值检测和处理
    - 历史缓冲区备用策略
    """
    
    def __init__(self, config: Optional[StabilizerConfig] = None):
        """
        初始化
        
        Args:
            config: 稳定器配置
        """
        self.config = config or StabilizerConfig()
        
        # 2D 滤波器 {keypoint_id: (filter_x, filter_y)}
        self._filters_2d: Dict[int, Tuple[OneEuroFilter, OneEuroFilter]] = {}
        
        # 3D 滤波器 {keypoint_id: (filter_x, filter_y, filter_z)}
        self._filters_3d: Dict[int, Tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter]] = {}
        
        # 历史缓冲区 {keypoint_id: KeypointHistory}
        self._history: Dict[int, KeypointHistory] = {}
    
    def _get_or_create_filter_2d(
        self,
        keypoint_id: int
    ) -> Tuple[OneEuroFilter, OneEuroFilter]:
        """获取或创建 2D 滤波器"""
        if keypoint_id not in self._filters_2d:
            cfg = self.config.filter_2d
            self._filters_2d[keypoint_id] = (
                OneEuroFilter(self.config.freq, cfg.mincutoff, cfg.beta, cfg.dcutoff),
                OneEuroFilter(self.config.freq, cfg.mincutoff, cfg.beta, cfg.dcutoff),
            )
        return self._filters_2d[keypoint_id]
    
    def _get_or_create_filter_3d(
        self,
        keypoint_id: int
    ) -> Tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter]:
        """获取或创建 3D 滤波器"""
        if keypoint_id not in self._filters_3d:
            cfg = self.config.filter_3d
            self._filters_3d[keypoint_id] = (
                OneEuroFilter(self.config.freq, cfg.mincutoff, cfg.beta, cfg.dcutoff),
                OneEuroFilter(self.config.freq, cfg.mincutoff, cfg.beta, cfg.dcutoff),
                OneEuroFilter(self.config.freq, cfg.mincutoff, cfg.beta, cfg.dcutoff),
            )
        return self._filters_3d[keypoint_id]
    
    def _get_or_create_history(self, keypoint_id: int) -> KeypointHistory:
        """获取或创建历史缓冲区"""
        if keypoint_id not in self._history:
            self._history[keypoint_id] = KeypointHistory(self.config.history_size)
        return self._history[keypoint_id]
    
    def _is_jump_2d(
        self,
        x: float,
        y: float,
        history: KeypointHistory
    ) -> bool:
        """检测 2D 跳变"""
        if len(history.values_2d) == 0:
            return False
        
        last = history.values_2d[-1]
        dx = abs(x - last[0])
        dy = abs(y - last[1])
        dist = math.sqrt(dx * dx + dy * dy)
        
        return dist > self.config.max_jump_2d
    
    def _is_jump_3d(
        self,
        x: float,
        y: float,
        z: float,
        history: KeypointHistory
    ) -> bool:
        """检测 3D 跳变"""
        if len(history.values_3d) == 0:
            return False
        
        last = history.values_3d[-1]
        dx = abs(x - last[0])
        dy = abs(y - last[1])
        dz = abs(z - last[2])
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        
        return dist > self.config.max_jump_3d
    
    def _get_fallback_2d(self, history: KeypointHistory) -> Optional[Tuple[float, float]]:
        """获取 2D 备用值（历史均值）"""
        if len(history.values_2d) == 0:
            return None
        
        xs = [v[0] for v in history.values_2d]
        ys = [v[1] for v in history.values_2d]
        return (np.mean(xs), np.mean(ys))
    
    def _get_fallback_3d(
        self,
        history: KeypointHistory
    ) -> Optional[Tuple[float, float, float]]:
        """获取 3D 备用值（历史均值）"""
        if len(history.values_3d) == 0:
            return None
        
        xs = [v[0] for v in history.values_3d]
        ys = [v[1] for v in history.values_3d]
        zs = [v[2] for v in history.values_3d]
        return (np.mean(xs), np.mean(ys), np.mean(zs))

    def stabilize_2d(
        self,
        keypoint_id: int,
        x: float,
        y: float,
        confidence: float,
        timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        稳定 2D 关键点
        
        Args:
            keypoint_id: 关键点 ID (0-32)
            x, y: 像素坐标
            confidence: 置信度 (0-1)
            timestamp: 时间戳（秒）
        
        Returns:
            (稳定后的 x, 稳定后的 y)
        """
        if not self.config.enable_2d_stabilization:
            return (x, y)
        
        if timestamp is None:
            timestamp = time.time()
        
        history = self._get_or_create_history(keypoint_id)
        
        # 置信度过低，使用历史值
        if confidence < self.config.low_confidence_threshold:
            fallback = self._get_fallback_2d(history)
            if fallback is not None:
                return fallback
            return (x, y)
        
        # 检测跳变
        if self._is_jump_2d(x, y, history):
            fallback = self._get_fallback_2d(history)
            if fallback is not None:
                # 使用历史值和当前值的加权平均
                weight = 0.3  # 当前值权重
                x = weight * x + (1 - weight) * fallback[0]
                y = weight * y + (1 - weight) * fallback[1]
        
        # 获取滤波器
        filter_x, filter_y = self._get_or_create_filter_2d(keypoint_id)
        
        # 置信度自适应：低置信度时增强滤波
        if confidence < self.config.medium_confidence_threshold:
            # 临时降低 mincutoff 增强平滑
            scale = confidence / self.config.medium_confidence_threshold
            filter_x.mincutoff = self.config.filter_2d.mincutoff * scale
            filter_y.mincutoff = self.config.filter_2d.mincutoff * scale
        else:
            filter_x.mincutoff = self.config.filter_2d.mincutoff
            filter_y.mincutoff = self.config.filter_2d.mincutoff
        
        # 滤波
        x_filtered = filter_x(x, timestamp)
        y_filtered = filter_y(y, timestamp)
        
        # 更新历史
        history.values_2d.append((x_filtered, y_filtered, confidence, timestamp))
        
        return (x_filtered, y_filtered)
    
    def stabilize_3d(
        self,
        keypoint_id: int,
        x: float,
        y: float,
        z: float,
        confidence: float,
        timestamp: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        稳定 3D 关键点
        
        Args:
            keypoint_id: 关键点 ID (0-32)
            x, y, z: 3D 坐标（米）
            confidence: 置信度 (0-1)
            timestamp: 时间戳（秒）
        
        Returns:
            (稳定后的 x, y, z)
        """
        if not self.config.enable_3d_stabilization:
            return (x, y, z)
        
        if timestamp is None:
            timestamp = time.time()
        
        history = self._get_or_create_history(keypoint_id)
        
        # 无效深度
        if z <= 0:
            fallback = self._get_fallback_3d(history)
            if fallback is not None:
                return fallback
            return (x, y, z)
        
        # 置信度过低，使用历史值
        if confidence < self.config.low_confidence_threshold:
            fallback = self._get_fallback_3d(history)
            if fallback is not None:
                return fallback
            return (x, y, z)
        
        # 检测跳变
        if self._is_jump_3d(x, y, z, history):
            fallback = self._get_fallback_3d(history)
            if fallback is not None:
                # 使用历史值和当前值的加权平均
                weight = 0.3
                x = weight * x + (1 - weight) * fallback[0]
                y = weight * y + (1 - weight) * fallback[1]
                z = weight * z + (1 - weight) * fallback[2]
        
        # 获取滤波器
        filter_x, filter_y, filter_z = self._get_or_create_filter_3d(keypoint_id)
        
        # 身高相关关键点的Y轴增强平滑
        HEIGHT_CRITICAL_KEYPOINTS = {0, 11, 12, 23, 24, 25, 26, 27, 28}
        y_extra_smooth = keypoint_id in HEIGHT_CRITICAL_KEYPOINTS
        
        # 置信度自适应
        if confidence < self.config.medium_confidence_threshold:
            scale = confidence / self.config.medium_confidence_threshold
            filter_x.mincutoff = self.config.filter_3d.mincutoff * scale
            filter_y.mincutoff = self.config.filter_3d.mincutoff * scale * (0.8 if y_extra_smooth else 1.0)
            filter_z.mincutoff = self.config.filter_3d.mincutoff * scale
        else:
            filter_x.mincutoff = self.config.filter_3d.mincutoff
            filter_y.mincutoff = self.config.filter_3d.mincutoff * (0.8 if y_extra_smooth else 1.0)
            filter_z.mincutoff = self.config.filter_3d.mincutoff
        
        # 滤波
        x_filtered = filter_x(x, timestamp)
        y_filtered = filter_y(y, timestamp)
        z_filtered = filter_z(z, timestamp)
        
        # 更新历史
        history.values_3d.append((x_filtered, y_filtered, z_filtered, confidence, timestamp))
        
        return (x_filtered, y_filtered, z_filtered)
    
    def stabilize_pose_2d(
        self,
        landmarks: List['Landmark'],
        image_width: int,
        image_height: int,
        timestamp: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """
        批量稳定 2D 姿态
        
        Args:
            landmarks: MediaPipe 关键点列表
            image_width: 图像宽度
            image_height: 图像高度
            timestamp: 时间戳
        
        Returns:
            稳定后的 2D 坐标列表 [(x, y), ...]
        """
        if timestamp is None:
            timestamp = time.time()
        
        result = []
        for i, lm in enumerate(landmarks):
            x = lm.x * image_width
            y = lm.y * image_height
            x_s, y_s = self.stabilize_2d(i, x, y, lm.visibility, timestamp)
            result.append((x_s, y_s))
        
        return result
    
    def stabilize_points_3d(
        self,
        points_3d: List['Point3D'],
        timestamp: Optional[float] = None
    ) -> List['Point3D']:
        """
        批量稳定 3D 点
        
        Args:
            points_3d: 3D 点列表
            timestamp: 时间戳
        
        Returns:
            稳定后的 3D 点列表
        """
        from src.core.coordinate_transformer import Point3D
        
        if timestamp is None:
            timestamp = time.time()
        
        result = []
        for i, pt in enumerate(points_3d):
            if pt.z <= 0 or pt.confidence <= 0:
                result.append(pt)
                continue
            
            x_s, y_s, z_s = self.stabilize_3d(
                i, pt.x, pt.y, pt.z, pt.confidence, timestamp
            )
            result.append(Point3D(x_s, y_s, z_s, pt.confidence))
        
        return result
    
    def reset(self, keypoint_id: Optional[int] = None):
        """
        重置滤波器
        
        Args:
            keypoint_id: 关键点 ID，如果为 None 则重置所有
        """
        if keypoint_id is not None:
            # 重置指定关键点
            if keypoint_id in self._filters_2d:
                for f in self._filters_2d[keypoint_id]:
                    f.reset()
            if keypoint_id in self._filters_3d:
                for f in self._filters_3d[keypoint_id]:
                    f.reset()
            if keypoint_id in self._history:
                self._history[keypoint_id] = KeypointHistory(self.config.history_size)
        else:
            # 重置所有
            for filters in self._filters_2d.values():
                for f in filters:
                    f.reset()
            for filters in self._filters_3d.values():
                for f in filters:
                    f.reset()
            self._history.clear()
