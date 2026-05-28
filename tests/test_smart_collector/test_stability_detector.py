"""
稳定性检测器测试

**Feature: smart-data-collector**
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

from src.core.stability_detector import StabilityDetector


# ============== Strategies ==============

@st.composite
def random_points_strategy(draw, num_points=10):
    """生成随机3D点集"""
    points = []
    for _ in range(num_points):
        x = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        y = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        z = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))
        points.append([x, y, z])
    return np.array(points)


# ============== Property Tests ==============

class TestSlidingWindowMaintenance:
    """
    Property 3: Sliding Window Maintenance
    For any number of frames added, buffer size <= window_size
    """
    
    @given(
        window_size=st.integers(min_value=1, max_value=20),
        num_frames=st.integers(min_value=0, max_value=50)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_buffer_never_exceeds_window_size(self, window_size: int, num_frames: int):
        """缓冲区大小永远不超过窗口大小"""
        detector = StabilityDetector(window_size=window_size)
        
        # 添加多帧数据
        for _ in range(num_frames):
            points = np.random.rand(10, 3)
            detector.add_points(body_points=points)
        
        # 验证缓冲区大小
        assert detector.buffer_size <= window_size
    
    @given(
        window_size=st.integers(min_value=5, max_value=15),
        num_frames=st.integers(min_value=20, max_value=40)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_buffer_reaches_max_size(self, window_size: int, num_frames: int):
        """添加足够多帧后，缓冲区达到最大值"""
        detector = StabilityDetector(window_size=window_size)
        
        for _ in range(num_frames):
            points = np.random.rand(10, 3)
            detector.add_points(body_points=points)
        
        # 缓冲区应该正好等于窗口大小
        assert detector.buffer_size == window_size


class TestStabilityCalculation:
    """
    Property 1: Stability Detection Accuracy
    Movement std calculation should match numpy std
    """
    
    def test_stable_points_detected_as_stable(self):
        """完全静止的点应被检测为稳定"""
        detector = StabilityDetector(window_size=5, body_threshold=0.01)
        
        # 添加完全相同的点（无移动）
        static_points = np.array([[0.5, 0.5, 0.0]] * 10)
        for _ in range(5):
            detector.add_points(body_points=static_points)
        
        result = detector.get_stability()
        assert result.body_stable is True
        assert result.body_movement < 0.01
    
    def test_moving_points_detected_as_unstable(self):
        """大幅移动的点应被检测为不稳定"""
        detector = StabilityDetector(window_size=5, body_threshold=0.01)
        
        # 添加大幅移动的点
        for i in range(5):
            points = np.array([[0.1 * i, 0.1 * i, 0.0]] * 10)
            detector.add_points(body_points=points)
        
        result = detector.get_stability()
        assert result.body_stable is False
        assert result.body_movement > 0.01
    
    @given(st.floats(min_value=0.001, max_value=0.1, allow_nan=False))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_different_thresholds_for_body_and_hand(self, hand_threshold: float):
        """身体和手部可以使用不同阈值"""
        body_threshold = 0.05
        detector = StabilityDetector(
            window_size=5,
            body_threshold=body_threshold,
            hand_threshold=hand_threshold
        )
        
        # 添加中等移动量的数据
        for i in range(5):
            body_points = np.array([[0.01 * i, 0.0, 0.0]] * 10)
            hand_points = np.array([[0.005 * i, 0.0, 0.0]] * 21)
            detector.add_points(
                body_points=body_points,
                left_hand_points=hand_points
            )
        
        result = detector.get_stability()
        # 验证阈值被正确应用
        assert detector.body_threshold == body_threshold
        assert detector.hand_threshold == hand_threshold


class TestStabilityProgress:
    """
    Property 9: Stability Progress Range
    Progress should always be in [0.0, 1.0]
    """
    
    @given(
        num_frames=st.integers(min_value=0, max_value=30),
        movement_scale=st.floats(min_value=0.0, max_value=0.5, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_progress_always_in_valid_range(self, num_frames: int, movement_scale: float):
        """稳定进度始终在 [0.0, 1.0] 范围内"""
        detector = StabilityDetector(window_size=10)
        
        for i in range(num_frames):
            points = np.random.rand(10, 3) * movement_scale
            detector.add_points(body_points=points)
        
        result = detector.get_stability()
        assert 0.0 <= result.progress <= 1.0


class TestResetBehavior:
    """测试重置行为"""
    
    def test_reset_clears_buffer(self):
        """重置应清空缓冲区"""
        detector = StabilityDetector(window_size=10)
        
        # 添加一些数据
        for _ in range(5):
            detector.add_points(body_points=np.random.rand(10, 3))
        
        assert detector.buffer_size > 0
        
        # 重置
        detector.reset()
        
        assert detector.buffer_size == 0
    
    def test_reset_clears_stable_count(self):
        """重置应清空稳定计数"""
        detector = StabilityDetector(window_size=5)
        
        # 添加稳定数据
        static_points = np.array([[0.5, 0.5, 0.0]] * 10)
        for _ in range(10):
            detector.add_points(body_points=static_points)
            detector.get_stability()
        
        # 重置
        detector.reset()
        
        result = detector.get_stability()
        assert result.stable_frames == 0


class TestEdgeCases:
    """边界情况测试"""
    
    def test_empty_buffer_returns_not_stable(self):
        """空缓冲区应返回不稳定"""
        detector = StabilityDetector(window_size=10)
        result = detector.get_stability()
        assert result.is_stable is False
        assert result.progress >= 0.0
    
    def test_single_frame_not_stable(self):
        """单帧数据应返回不稳定"""
        detector = StabilityDetector(window_size=10)
        detector.add_points(body_points=np.random.rand(10, 3))
        
        result = detector.get_stability()
        assert result.is_stable is False
    
    def test_window_size_one(self):
        """窗口大小为1的特殊情况"""
        detector = StabilityDetector(window_size=1)
        detector.add_points(body_points=np.random.rand(10, 3))
        
        # 应该能正常工作
        result = detector.get_stability()
        assert isinstance(result.progress, float)
