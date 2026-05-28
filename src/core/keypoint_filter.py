"""
关键点滤波器
对3D关键点进行时序滤波，提高稳定性
"""
from collections import deque
from typing import List, Optional

import numpy as np

from .coordinate_transformer import Point3D
from ..utils.logger import get_logger

logger = get_logger(__name__)


class KeypointFilter:
    """
    关键点时序滤波器
    
    使用滑动窗口对3D关键点进行平滑处理
    
    Example:
        >>> filter = KeypointFilter(window_size=5)
        >>> smoothed = filter.update(points_3d)
        >>> stability = filter.get_stability_score()
    """
    
    def __init__(
        self,
        window_size: int = 5,
        num_landmarks: int = 33,
        use_weighted: bool = True
    ):
        """
        初始化滤波器
        
        Args:
            window_size: 滑动窗口大小
            num_landmarks: 关键点数量
            use_weighted: 是否使用加权平均（越近的帧权重越大）
        """
        self.window_size = window_size
        self.num_landmarks = num_landmarks
        self.use_weighted = use_weighted
        
        # 历史数据队列
        self._history: deque = deque(maxlen=window_size)
        
        # 生成权重（指数衰减）
        if use_weighted:
            weights = np.exp(np.linspace(-1, 0, window_size))
            self._weights = weights / weights.sum()
        else:
            self._weights = np.ones(window_size) / window_size
        
        logger.debug(f"KeypointFilter 初始化: window={window_size}, weighted={use_weighted}")
    
    def update(self, points: List[Point3D]) -> List[Point3D]:
        """
        更新滤波器并返回平滑后的关键点
        
        Args:
            points: 当前帧的3D关键点列表
        
        Returns:
            平滑后的3D关键点列表
        """
        if len(points) != self.num_landmarks:
            logger.warning(f"关键点数量不匹配: {len(points)} vs {self.num_landmarks}")
        
        # 转换为数组存储
        points_array = np.array([[p.x, p.y, p.z, p.confidence] for p in points])
        self._history.append(points_array)
        
        # 如果历史数据不足，直接返回当前点
        if len(self._history) < 2:
            return points
        
        # 计算加权平均
        smoothed = self._compute_smoothed()
        
        # 转换回 Point3D 列表
        result = []
        for i in range(len(points)):
            result.append(Point3D(
                x=smoothed[i, 0],
                y=smoothed[i, 1],
                z=smoothed[i, 2],
                confidence=smoothed[i, 3]
            ))
        
        return result
    
    def _compute_smoothed(self) -> np.ndarray:
        """计算加权平均"""
        history_list = list(self._history)
        n = len(history_list)
        
        # 使用最近的权重
        weights = self._weights[-n:]
        weights = weights / weights.sum()  # 重新归一化
        
        # 加权平均
        smoothed = np.zeros_like(history_list[0])
        for i, data in enumerate(history_list):
            smoothed += weights[i] * data
        
        return smoothed
    
    def reset(self) -> None:
        """重置滤波器状态"""
        self._history.clear()
        logger.debug("KeypointFilter 已重置")
    
    def get_stability_score(self) -> float:
        """
        计算当前稳定性评分
        
        基于历史数据的标准差，标准差越小越稳定
        
        Returns:
            稳定性评分 [0, 1]，1表示最稳定
        """
        if len(self._history) < 2:
            return 0.0
        
        history_array = np.array(list(self._history))
        
        # 计算XYZ坐标的标准差
        std = np.std(history_array[:, :, :3], axis=0)
        mean_std = np.mean(std)
        
        # 转换为稳定性评分（标准差越小越稳定）
        # 假设标准差 0.1m 对应稳定性 0
        stability = max(0.0, 1.0 - mean_std * 10)
        
        return stability
    
    def get_velocity(self) -> Optional[List[np.ndarray]]:
        """
        获取关键点速度（用于动作分析）
        
        Returns:
            每个关键点的速度向量列表，或 None（历史数据不足）
        """
        if len(self._history) < 2:
            return None
        
        current = self._history[-1]
        previous = self._history[-2]
        
        velocities = []
        for i in range(self.num_landmarks):
            v = current[i, :3] - previous[i, :3]
            velocities.append(v)
        
        return velocities
    
    @property
    def is_ready(self) -> bool:
        """滤波器是否已准备好（有足够的历史数据）"""
        return len(self._history) >= 2
    
    @property
    def history_length(self) -> int:
        """当前历史数据长度"""
        return len(self._history)


class OneEuroFilter:
    """
    One Euro Filter - 更高级的自适应滤波器
    
    参考论文: "1€ Filter: A Simple Speed-based Low-pass Filter for 
    Noisy Input in Interactive Systems"
    
    特点：
    - 低速时平滑更强
    - 高速时响应更快
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        """
        初始化 One Euro Filter
        
        Args:
            min_cutoff: 最小截止频率
            beta: 速度系数（越大，对速度变化越敏感）
            d_cutoff: 导数的截止频率
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None
    
    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        """计算平滑因子"""
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)
    
    def _exponential_smoothing(self, a: float, x: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        """指数平滑"""
        return a * x + (1 - a) * x_prev
    
    def update(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        更新滤波器
        
        Args:
            x: 当前值
            t: 当前时间戳
        
        Returns:
            滤波后的值
        """
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x
        
        t_e = t - self._t_prev
        if t_e <= 0:
            return self._x_prev
        
        # 计算导数
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self._dx_prev)
        
        # 计算自适应截止频率
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        
        # 平滑
        a = self._smoothing_factor(t_e, np.mean(cutoff))
        x_hat = self._exponential_smoothing(a, x, self._x_prev)
        
        # 更新状态
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        
        return x_hat
    
    def reset(self) -> None:
        """重置滤波器"""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None


class KeypointFilterAdvanced:
    """
    高级关键点滤波器
    
    为每个关键点使用独立的 One Euro Filter
    """
    
    def __init__(
        self,
        num_landmarks: int = 33,
        min_cutoff: float = 1.0,
        beta: float = 0.007
    ):
        """
        初始化高级滤波器
        
        Args:
            num_landmarks: 关键点数量
            min_cutoff: 最小截止频率
            beta: 速度系数
        """
        self.num_landmarks = num_landmarks
        self._filters = [
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            for _ in range(num_landmarks)
        ]
    
    def update(self, points: List[Point3D], timestamp: float) -> List[Point3D]:
        """
        更新滤波器
        
        Args:
            points: 当前帧的3D关键点
            timestamp: 时间戳
        
        Returns:
            滤波后的3D关键点
        """
        result = []
        
        for i, point in enumerate(points):
            if i < len(self._filters):
                x = np.array([point.x, point.y, point.z])
                x_filtered = self._filters[i].update(x, timestamp)
                result.append(Point3D(
                    x=x_filtered[0],
                    y=x_filtered[1],
                    z=x_filtered[2],
                    confidence=point.confidence
                ))
            else:
                result.append(point)
        
        return result
    
    def reset(self) -> None:
        """重置所有滤波器"""
        for f in self._filters:
            f.reset()
