"""
稳定性检测器

判定标准：
1. 全身33个骨骼关键点全部可见 → 全身完整
2. 手腕(15,16)和脚踝(27,28)在2秒窗口内无大幅移动 → 站稳
3. 两者同时满足 → is_stable = True
"""
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .smart_collector_types import StabilityResult
from .hand_result import HolisticResult
from ..utils.logger import get_logger

logger = get_logger(__name__)

# 手腕 + 脚踝 关键点 ID
HAND_FOOT_IDS = [15, 16, 27, 28]

# 全身33个关键点
ALL_POSE_IDS = list(range(33))


class StabilityDetector:
    """
    稳定性检测器

    判定逻辑：
    1. 全身33个骨骼点全部可见（visibility >= min_confidence）
    2. 手腕和脚踝在 window_size 帧内移动标准差 < movement_threshold
    3. 两者同时满足 → 稳定
    """

    def __init__(
        self,
        window_size: int = 60,
        movement_threshold: float = 0.012,
        min_confidence: float = 0.5
    ):
        """
        Args:
            window_size: 检测窗口大小（帧数），60帧=2秒@30fps
            movement_threshold: 手脚移动标准差阈值（米），低于此值视为静止
            min_confidence: 关键点最小置信度
        """
        self.window_size = window_size
        self.movement_threshold = movement_threshold
        self.min_confidence = min_confidence

        # 手腕+脚踝坐标缓冲区：{keypoint_id: deque of (x,y,z)}
        self._buffers: dict = {kid: deque(maxlen=window_size) for kid in HAND_FOOT_IDS}

        # 全身骨骼可见性标记
        self._full_skeleton = False
        self._stable_count = 0

        logger.debug(f"StabilityDetector: window={window_size}, threshold={movement_threshold}")

    def add_frame(self, holistic_result: HolisticResult) -> None:
        """添加一帧数据"""
        if not holistic_result.pose.detected or not holistic_result.pose.landmarks:
            self._full_skeleton = False
            return

        landmarks = holistic_result.pose.landmarks

        # 1. 检查全身骨骼是否基本完整（低门槛：只要关键点存在即可）
        visible_count = sum(
            1 for i in range(min(33, len(landmarks)))
            if landmarks[i].visibility >= 0.3  # 存在性检查用低门槛
        )
        self._full_skeleton = (visible_count >= 25)  # 25/33 点可见即可

        # 2. 提取手腕+脚踝坐标（坐标精度用较高门槛）
        for kid in HAND_FOOT_IDS:
            if kid < len(landmarks) and landmarks[kid].visibility >= 0.3:
                lm = landmarks[kid]
                self._buffers[kid].append((lm.x, lm.y, lm.z))
            else:
                # 该点不可见，插入 None 占位保持时间对齐
                self._buffers[kid].append(None)

    def get_stability(self) -> StabilityResult:
        """
        获取当前稳定性状态

        Returns:
            StabilityResult
        """
        # 检查缓冲区是否足够
        has_enough_data = all(
            len(buf) >= max(10, self.window_size // 3) for buf in self._buffers.values()
        )

        if not self._full_skeleton or not has_enough_data:
            return StabilityResult(
                is_stable=False,
                progress=self._calculate_progress(),
                body_stable=False,
                left_hand_stable=False,
                right_hand_stable=False,
                body_movement=-1,
                hand_movement=-1,
                stable_frames=0
            )

        # 计算每个手脚关键点的移动标准差
        movements = {}
        for kid in HAND_FOOT_IDS:
            movements[kid] = self._point_movement(self._buffers[kid])

        # 手腕稳定性：取左右手腕的最大移动量
        hand_movement = max(
            movements.get(15, -1) if movements.get(15, -1) >= 0 else 0,
            movements.get(16, -1) if movements.get(16, -1) >= 0 else 0
        )
        # 脚踝稳定性：取左右脚踝的最大移动量
        foot_movement = max(
            movements.get(27, -1) if movements.get(27, -1) >= 0 else 0,
            movements.get(28, -1) if movements.get(28, -1) >= 0 else 0
        )

        hand_stable = hand_movement < self.movement_threshold
        foot_stable = foot_movement < self.movement_threshold
        is_stable = hand_stable and foot_stable

        if is_stable:
            self._stable_count += 1
        else:
            self._stable_count = 0

        combined_movement = max(hand_movement, foot_movement)

        return StabilityResult(
            is_stable=is_stable,
            progress=self._calculate_progress(),
            body_stable=foot_stable,      # 脚踝稳定 → 身体稳定
            left_hand_stable=movements.get(15, -1) < self.movement_threshold,
            right_hand_stable=movements.get(16, -1) < self.movement_threshold,
            body_movement=foot_movement,
            hand_movement=hand_movement,
            stable_frames=self._stable_count
        )

    def _point_movement(self, buffer: deque) -> float:
        """计算单个关键点在窗口内的移动标准差"""
        valid = [p for p in buffer if p is not None]
        if len(valid) < 2:
            return -1.0
        arr = np.array(valid)  # (N, 3)
        std_per_axis = np.std(arr, axis=0)  # (3,)
        return float(np.mean(std_per_axis))

    def _calculate_progress(self) -> float:
        """计算稳定进度 0.0-1.0"""
        # 缓冲区填充比例
        fill_ratio = min(
            len(buf) / self.window_size for buf in self._buffers.values()
        )

        if not self._full_skeleton:
            return fill_ratio * 0.3  # 骨骼不完整最多30%

        # 计算手脚移动量
        movements = []
        for kid in HAND_FOOT_IDS:
            m = self._point_movement(self._buffers[kid])
            if m >= 0:
                movements.append(m)

        if not movements:
            return fill_ratio * 0.5

        avg_movement = np.mean(movements)
        # 移动量越小，进度越高
        movement_ratio = max(0, 1 - avg_movement / self.movement_threshold)

        return min(1.0, fill_ratio * movement_ratio)

    @property
    def is_full_skeleton(self) -> bool:
        """全身骨骼是否完整"""
        return self._full_skeleton

    @property
    def is_buffer_full(self) -> bool:
        """缓冲区是否已满"""
        return all(len(buf) >= self.window_size for buf in self._buffers.values())

    @property
    def buffer_size(self) -> int:
        """当前缓冲区大小"""
        return min(len(buf) for buf in self._buffers.values())

    def reset(self) -> None:
        """重置检测器"""
        for buf in self._buffers.values():
            buf.clear()
        self._full_skeleton = False
        self._stable_count = 0
        logger.debug("StabilityDetector 已重置")
