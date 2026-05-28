"""
手部关键点滤波器和跟踪器

实现卡尔曼滤波平滑、轨迹预测、跟踪丢失恢复和双手ID一致性维护

**Feature: holistic-hand-integration**
**Properties: 7, 8, 9**
"""
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Deque

import numpy as np

from .coordinate_transformer import Point3D
from .hand_result import HandResult, HolisticResult
from .constants import HAND_LANDMARK_COUNT
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KalmanState:
    """单个关键点的卡尔曼滤波状态"""
    # 状态向量: [x, y, z, vx, vy, vz]
    x: np.ndarray  # 状态估计
    P: np.ndarray  # 协方差矩阵
    
    @classmethod
    def create(cls, initial_pos: np.ndarray) -> 'KalmanState':
        """创建初始状态"""
        x = np.zeros(6)
        x[:3] = initial_pos
        P = np.eye(6) * 0.1
        return cls(x=x, P=P)


class HandKeypointFilter:
    """
    手部关键点滤波器
    
    使用卡尔曼滤波平滑手部关键点轨迹，支持跟踪丢失预测和重置
    
    **Property 7: 滤波器轨迹平滑性**
    **Property 8: 跟踪丢失恢复**
    
    Example:
        >>> filter = HandKeypointFilter(window_size=5, max_lost_frames=5)
        >>> smoothed, is_predicted = filter.update(hand_result, timestamp)
    """
    
    # 卡尔曼滤波参数
    PROCESS_NOISE = 0.01      # 过程噪声
    MEASUREMENT_NOISE = 0.005  # 测量噪声
    
    def __init__(
        self,
        window_size: int = 5,
        max_lost_frames: int = 5,
        use_kalman: bool = True
    ):
        """
        初始化滤波器
        
        Args:
            window_size: 滑动窗口大小（用于简单平均滤波）
            max_lost_frames: 最大允许丢失帧数，超过则重置
            use_kalman: 是否使用卡尔曼滤波
        """
        self.window_size = window_size
        self.max_lost_frames = max_lost_frames
        self.use_kalman = use_kalman
        
        # 状态
        self._lost_frames = 0
        self._initialized = False
        self._last_timestamp = 0.0
        
        # 卡尔曼滤波状态 (21个关键点)
        self._kalman_states: List[Optional[KalmanState]] = [None] * HAND_LANDMARK_COUNT
        
        # 滑动窗口历史 (用于简单滤波或备用)
        self._history: Deque[List[Point3D]] = deque(maxlen=window_size)
        
        # 最后有效的3D点
        self._last_valid_points: Optional[List[Point3D]] = None
    
    @property
    def lost_frames(self) -> int:
        """连续丢失帧数"""
        return self._lost_frames
    
    @property
    def is_tracking(self) -> bool:
        """是否正在跟踪"""
        return self._initialized and self._lost_frames < self.max_lost_frames
    
    def update(
        self,
        points_3d: Optional[List[Point3D]],
        timestamp: float
    ) -> Tuple[List[Point3D], bool]:
        """
        更新滤波器
        
        Args:
            points_3d: 21个3D关键点，None表示未检测到
            timestamp: 时间戳
        
        Returns:
            (滤波后的3D点, 是否为预测值)
        """
        dt = timestamp - self._last_timestamp if self._last_timestamp > 0 else 0.033
        self._last_timestamp = timestamp
        
        # 未检测到手部
        if points_3d is None or len(points_3d) == 0:
            self._lost_frames += 1
            
            # 超过最大丢失帧数，重置
            if self._lost_frames >= self.max_lost_frames:
                self.reset()
                return [], True
            
            # 预测位置
            if self._initialized:
                predicted = self.predict_position(dt)
                return predicted, True
            
            return [], True
        
        # 检测到手部，重置丢失计数
        self._lost_frames = 0
        
        # 首次初始化
        if not self._initialized:
            self._initialize(points_3d)
            self._last_valid_points = points_3d
            return points_3d, False
        
        # 应用滤波
        if self.use_kalman:
            filtered = self._kalman_update(points_3d, dt)
        else:
            filtered = self._simple_filter(points_3d)
        
        self._last_valid_points = filtered
        return filtered, False
    
    def predict_position(self, dt: float = 0.033) -> List[Point3D]:
        """
        线性外推预测手部位置
        
        **Property 8: 跟踪丢失恢复**
        
        Args:
            dt: 时间间隔
        
        Returns:
            预测的21个3D点
        """
        if not self._initialized:
            return []
        
        predicted = []
        
        for i, state in enumerate(self._kalman_states):
            if state is not None:
                # 使用速度进行线性预测
                pos = state.x[:3] + state.x[3:6] * dt
                # 降低置信度
                confidence = 0.5 * max(0, 1 - self._lost_frames / self.max_lost_frames)
                predicted.append(Point3D(
                    x=pos[0], y=pos[1], z=pos[2],
                    confidence=confidence
                ))
            elif self._last_valid_points and i < len(self._last_valid_points):
                # 使用最后有效点
                p = self._last_valid_points[i]
                predicted.append(Point3D(
                    x=p.x, y=p.y, z=p.z,
                    confidence=0.3
                ))
            else:
                predicted.append(Point3D(0, 0, 0, confidence=0.0))
        
        return predicted
    
    def reset(self) -> None:
        """重置滤波器状态"""
        self._lost_frames = 0
        self._initialized = False
        self._last_timestamp = 0.0
        self._kalman_states = [None] * HAND_LANDMARK_COUNT
        self._history.clear()
        self._last_valid_points = None
        logger.debug("HandKeypointFilter 已重置")
    
    def _initialize(self, points_3d: List[Point3D]) -> None:
        """初始化滤波器状态"""
        for i, point in enumerate(points_3d):
            if point.is_valid():
                self._kalman_states[i] = KalmanState.create(
                    np.array([point.x, point.y, point.z])
                )
        
        self._history.append(points_3d)
        self._initialized = True
        logger.debug("HandKeypointFilter 已初始化")
    
    def _kalman_update(self, points_3d: List[Point3D], dt: float) -> List[Point3D]:
        """
        卡尔曼滤波更新
        
        **Property 7: 滤波器轨迹平滑性**
        """
        filtered = []
        
        # 状态转移矩阵
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # 过程噪声
        Q = np.eye(6) * self.PROCESS_NOISE
        Q[3:, 3:] *= 10  # 速度噪声更大
        
        # 测量矩阵
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        
        # 测量噪声
        R = np.eye(3) * self.MEASUREMENT_NOISE
        
        for i, point in enumerate(points_3d):
            state = self._kalman_states[i]
            
            if not point.is_valid():
                # 无效测量，仅预测
                if state is not None:
                    state.x = F @ state.x
                    state.P = F @ state.P @ F.T + Q
                    filtered.append(Point3D(
                        x=state.x[0], y=state.x[1], z=state.x[2],
                        confidence=point.confidence * 0.8
                    ))
                else:
                    filtered.append(point)
                continue
            
            measurement = np.array([point.x, point.y, point.z])
            
            if state is None:
                # 新初始化
                self._kalman_states[i] = KalmanState.create(measurement)
                filtered.append(point)
                continue
            
            # 预测步骤
            x_pred = F @ state.x
            P_pred = F @ state.P @ F.T + Q
            
            # 更新步骤
            y = measurement - H @ x_pred  # 残差
            S = H @ P_pred @ H.T + R      # 残差协方差
            K = P_pred @ H.T @ np.linalg.inv(S)  # 卡尔曼增益
            
            state.x = x_pred + K @ y
            state.P = (np.eye(6) - K @ H) @ P_pred
            
            filtered.append(Point3D(
                x=state.x[0], y=state.x[1], z=state.x[2],
                confidence=point.confidence
            ))
        
        self._history.append(filtered)
        return filtered
    
    def _simple_filter(self, points_3d: List[Point3D]) -> List[Point3D]:
        """简单滑动窗口平均滤波"""
        self._history.append(points_3d)
        
        if len(self._history) < 2:
            return points_3d
        
        filtered = []
        for i in range(len(points_3d)):
            valid_points = []
            for frame in self._history:
                if i < len(frame) and frame[i].is_valid():
                    valid_points.append(frame[i])
            
            if valid_points:
                avg_x = sum(p.x for p in valid_points) / len(valid_points)
                avg_y = sum(p.y for p in valid_points) / len(valid_points)
                avg_z = sum(p.z for p in valid_points) / len(valid_points)
                avg_conf = sum(p.confidence for p in valid_points) / len(valid_points)
                filtered.append(Point3D(avg_x, avg_y, avg_z, avg_conf))
            else:
                filtered.append(points_3d[i])
        
        return filtered



class HandTracker:
    """
    双手跟踪器
    
    维护左右手ID一致性，防止左右手交换
    
    **Property 9: 手部ID一致性**
    
    Example:
        >>> tracker = HandTracker()
        >>> left_3d, right_3d = tracker.update(holistic_result, points_3d_left, points_3d_right, timestamp)
    """
    
    # 手腕位置历史长度
    WRIST_HISTORY_LENGTH = 10
    
    # 手部交换检测阈值（米）
    SWAP_DISTANCE_THRESHOLD = 0.15
    
    def __init__(self):
        """初始化跟踪器"""
        self.left_filter = HandKeypointFilter()
        self.right_filter = HandKeypointFilter()
        
        # 手腕位置历史
        self._left_wrist_history: Deque[Point3D] = deque(maxlen=self.WRIST_HISTORY_LENGTH)
        self._right_wrist_history: Deque[Point3D] = deque(maxlen=self.WRIST_HISTORY_LENGTH)
        
        # 上一帧的手部检测状态
        self._last_left_detected = False
        self._last_right_detected = False
    
    def update(
        self,
        left_points_3d: Optional[List[Point3D]],
        right_points_3d: Optional[List[Point3D]],
        timestamp: float
    ) -> Tuple[Optional[List[Point3D]], Optional[List[Point3D]]]:
        """
        更新双手跟踪
        
        **Property 9: 手部ID一致性**
        
        Args:
            left_points_3d: 左手3D关键点
            right_points_3d: 右手3D关键点
            timestamp: 时间戳
        
        Returns:
            (左手3D点, 右手3D点)，未检测到的手返回None
        """
        # 检测手部交换
        left_points_3d, right_points_3d = self._resolve_hand_identity(
            left_points_3d, right_points_3d
        )
        
        # 更新手腕历史
        self._update_wrist_history(left_points_3d, right_points_3d)
        
        # 应用滤波
        left_filtered = None
        right_filtered = None
        
        if left_points_3d is not None and len(left_points_3d) > 0:
            left_filtered, _ = self.left_filter.update(left_points_3d, timestamp)
            if len(left_filtered) == 0:
                left_filtered = None
        else:
            # 尝试预测
            predicted, is_predicted = self.left_filter.update(None, timestamp)
            if is_predicted and len(predicted) > 0:
                left_filtered = predicted
        
        if right_points_3d is not None and len(right_points_3d) > 0:
            right_filtered, _ = self.right_filter.update(right_points_3d, timestamp)
            if len(right_filtered) == 0:
                right_filtered = None
        else:
            # 尝试预测
            predicted, is_predicted = self.right_filter.update(None, timestamp)
            if is_predicted and len(predicted) > 0:
                right_filtered = predicted
        
        # 更新检测状态
        self._last_left_detected = left_filtered is not None
        self._last_right_detected = right_filtered is not None
        
        return left_filtered, right_filtered
    
    def _resolve_hand_identity(
        self,
        left_points: Optional[List[Point3D]],
        right_points: Optional[List[Point3D]]
    ) -> Tuple[Optional[List[Point3D]], Optional[List[Point3D]]]:
        """
        解决左右手ID交换问题
        
        基于手腕位置历史判断是否发生了左右手交换
        
        **Property 9: 手部ID一致性**
        
        Args:
            left_points: 当前帧标记为左手的点
            right_points: 当前帧标记为右手的点
        
        Returns:
            校正后的 (左手点, 右手点)
        """
        # 如果历史不足，无法判断交换
        if len(self._left_wrist_history) < 3 or len(self._right_wrist_history) < 3:
            return left_points, right_points
        
        # 如果只有一只手，无需判断
        if left_points is None or right_points is None:
            return left_points, right_points
        
        if len(left_points) == 0 or len(right_points) == 0:
            return left_points, right_points
        
        # 获取当前手腕位置
        current_left_wrist = left_points[0]  # 手腕是索引0
        current_right_wrist = right_points[0]
        
        if not current_left_wrist.is_valid() or not current_right_wrist.is_valid():
            return left_points, right_points
        
        # 计算历史平均手腕位置
        avg_left_wrist = self._get_average_wrist(self._left_wrist_history)
        avg_right_wrist = self._get_average_wrist(self._right_wrist_history)
        
        if avg_left_wrist is None or avg_right_wrist is None:
            return left_points, right_points
        
        # 计算当前标记与历史的距离
        dist_left_to_left = current_left_wrist.distance_to(avg_left_wrist)
        dist_left_to_right = current_left_wrist.distance_to(avg_right_wrist)
        dist_right_to_left = current_right_wrist.distance_to(avg_left_wrist)
        dist_right_to_right = current_right_wrist.distance_to(avg_right_wrist)
        
        # 判断是否需要交换
        # 如果当前"左手"更接近历史"右手"，且当前"右手"更接近历史"左手"
        should_swap = (
            dist_left_to_right < dist_left_to_left and
            dist_right_to_left < dist_right_to_right and
            dist_left_to_right < self.SWAP_DISTANCE_THRESHOLD and
            dist_right_to_left < self.SWAP_DISTANCE_THRESHOLD
        )
        
        if should_swap:
            logger.debug("检测到左右手交换，进行校正")
            return right_points, left_points
        
        return left_points, right_points
    
    def _update_wrist_history(
        self,
        left_points: Optional[List[Point3D]],
        right_points: Optional[List[Point3D]]
    ) -> None:
        """更新手腕位置历史"""
        if left_points is not None and len(left_points) > 0:
            wrist = left_points[0]
            if wrist.is_valid():
                self._left_wrist_history.append(wrist)
        
        if right_points is not None and len(right_points) > 0:
            wrist = right_points[0]
            if wrist.is_valid():
                self._right_wrist_history.append(wrist)
    
    def _get_average_wrist(self, history: Deque[Point3D]) -> Optional[Point3D]:
        """计算手腕位置历史平均值"""
        valid_points = [p for p in history if p.is_valid()]
        
        if len(valid_points) < 3:
            return None
        
        avg_x = sum(p.x for p in valid_points) / len(valid_points)
        avg_y = sum(p.y for p in valid_points) / len(valid_points)
        avg_z = sum(p.z for p in valid_points) / len(valid_points)
        
        return Point3D(avg_x, avg_y, avg_z, confidence=1.0)
    
    def reset(self) -> None:
        """重置跟踪器"""
        self.left_filter.reset()
        self.right_filter.reset()
        self._left_wrist_history.clear()
        self._right_wrist_history.clear()
        self._last_left_detected = False
        self._last_right_detected = False
        logger.debug("HandTracker 已重置")
    
    @property
    def left_lost_frames(self) -> int:
        """左手连续丢失帧数"""
        return self.left_filter.lost_frames
    
    @property
    def right_lost_frames(self) -> int:
        """右手连续丢失帧数"""
        return self.right_filter.lost_frames
    
    @property
    def is_tracking_left(self) -> bool:
        """是否正在跟踪左手"""
        return self.left_filter.is_tracking
    
    @property
    def is_tracking_right(self) -> bool:
        """是否正在跟踪右手"""
        return self.right_filter.is_tracking
