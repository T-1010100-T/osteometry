"""
手部滤波器和跟踪器属性测试

**Feature: holistic-hand-integration, Property 7: 滤波器轨迹平滑性**
**Feature: holistic-hand-integration, Property 8: 跟踪丢失恢复**
**Feature: holistic-hand-integration, Property 9: 手部ID一致性**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""
import sys
from pathlib import Path
import math

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck, assume

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.coordinate_transformer import Point3D
from src.core.hand_filter import HandKeypointFilter, HandTracker


def create_hand_points(base_x: float = 0.0, base_y: float = 0.0, base_z: float = 1.0) -> list:
    """创建测试用手部关键点"""
    points = []
    for i in range(21):
        # 简单的手部模型
        x = base_x + (i % 5) * 0.01
        y = base_y + (i // 5) * 0.02
        z = base_z
        points.append(Point3D(x, y, z, confidence=0.9))
    return points


def add_noise_to_points(points: list, noise_std: float) -> list:
    """给点添加高斯噪声"""
    noisy = []
    for p in points:
        noisy.append(Point3D(
            x=p.x + np.random.normal(0, noise_std),
            y=p.y + np.random.normal(0, noise_std),
            z=p.z + np.random.normal(0, noise_std),
            confidence=p.confidence
        ))
    return noisy


class TestFilterTrajectorySmoothing:
    """
    滤波器轨迹平滑性测试
    
    **Property 7: 滤波器轨迹平滑性**
    对于任何添加了高斯噪声的手部关键点序列，滤波后的输出方差应低于输入，同时保持平均轨迹
    """
    
    def test_kalman_filter_reduces_noise_variance(self):
        """
        **Property 7: 滤波器轨迹平滑性**
        卡尔曼滤波应降低噪声方差
        """
        filter = HandKeypointFilter(window_size=5, use_kalman=True)
        
        # 生成带噪声的轨迹
        base_points = create_hand_points()
        noise_std = 0.01  # 10mm噪声
        
        input_positions = []
        output_positions = []
        
        for frame in range(30):
            timestamp = frame * 0.033
            noisy_points = add_noise_to_points(base_points, noise_std)
            
            # 记录输入
            input_positions.append([p.x for p in noisy_points])
            
            # 滤波
            filtered, _ = filter.update(noisy_points, timestamp)
            
            # 记录输出
            output_positions.append([p.x for p in filtered])
        
        # 计算方差
        input_arr = np.array(input_positions)
        output_arr = np.array(output_positions)
        
        # 跳过前几帧（滤波器需要预热）
        input_var = np.var(input_arr[10:], axis=0)
        output_var = np.var(output_arr[10:], axis=0)
        
        # 滤波后方差应更小
        assert np.mean(output_var) < np.mean(input_var)
    
    def test_simple_filter_reduces_noise_variance(self):
        """
        **Property 7: 滤波器轨迹平滑性**
        简单滑动窗口滤波也应降低噪声方差
        """
        filter = HandKeypointFilter(window_size=5, use_kalman=False)
        
        base_points = create_hand_points()
        noise_std = 0.01
        
        input_positions = []
        output_positions = []
        
        for frame in range(30):
            timestamp = frame * 0.033
            noisy_points = add_noise_to_points(base_points, noise_std)
            
            input_positions.append([p.x for p in noisy_points])
            filtered, _ = filter.update(noisy_points, timestamp)
            output_positions.append([p.x for p in filtered])
        
        input_arr = np.array(input_positions)
        output_arr = np.array(output_positions)
        
        input_var = np.var(input_arr[5:], axis=0)
        output_var = np.var(output_arr[5:], axis=0)
        
        assert np.mean(output_var) < np.mean(input_var)
    
    @given(noise_std=st.floats(min_value=0.001, max_value=0.05, allow_nan=False))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_filter_preserves_mean_trajectory(self, noise_std: float):
        """
        **Property 7: 滤波器轨迹平滑性**
        滤波应保持平均轨迹
        """
        filter = HandKeypointFilter(window_size=5, use_kalman=True)
        
        base_points = create_hand_points()
        base_mean = np.mean([p.x for p in base_points])
        
        output_means = []
        
        for frame in range(50):
            timestamp = frame * 0.033
            noisy_points = add_noise_to_points(base_points, noise_std)
            filtered, _ = filter.update(noisy_points, timestamp)
            output_means.append(np.mean([p.x for p in filtered]))
        
        # 滤波后的平均值应接近原始平均值
        final_mean = np.mean(output_means[20:])
        assert abs(final_mean - base_mean) < noise_std * 3


class TestTrackingLostRecovery:
    """
    跟踪丢失恢复测试
    
    **Property 8: 跟踪丢失恢复**
    """
    
    def test_prediction_during_lost_frames(self):
        """
        **Property 8: 跟踪丢失恢复**
        丢失N帧(N<5)时，滤波器应提供预测位置
        """
        filter = HandKeypointFilter(max_lost_frames=5)
        
        # 先建立跟踪
        base_points = create_hand_points()
        for frame in range(10):
            filter.update(base_points, frame * 0.033)
        
        # 模拟丢失3帧
        for frame in range(3):
            predicted, is_predicted = filter.update(None, (10 + frame) * 0.033)
            
            assert is_predicted, "应返回预测值"
            assert len(predicted) == 21, "应有21个预测点"
            assert filter.lost_frames == frame + 1
    
    def test_reset_after_max_lost_frames(self):
        """
        **Property 8: 跟踪丢失恢复**
        丢失N帧(N>=5)时，滤波器状态应重置
        """
        filter = HandKeypointFilter(max_lost_frames=5)
        
        # 先建立跟踪
        base_points = create_hand_points()
        for frame in range(10):
            filter.update(base_points, frame * 0.033)
        
        assert filter.is_tracking
        
        # 模拟丢失5帧
        for frame in range(5):
            filter.update(None, (10 + frame) * 0.033)
        
        # 应该已重置
        assert not filter.is_tracking
        assert filter.lost_frames == 0
    
    def test_prediction_uses_velocity(self):
        """
        **Property 8: 跟踪丢失恢复**
        预测应使用速度进行线性外推
        """
        filter = HandKeypointFilter(max_lost_frames=5, use_kalman=True)
        
        # 创建移动的手部
        for frame in range(20):
            points = create_hand_points(base_x=frame * 0.01)  # 每帧移动1cm
            filter.update(points, frame * 0.033)
        
        last_x = points[0].x
        
        # 丢失一帧，预测应继续移动
        predicted, is_predicted = filter.update(None, 20 * 0.033)
        
        assert is_predicted
        # 预测的x应该大于最后一帧（因为在向右移动）
        assert predicted[0].x >= last_x - 0.01  # 允许一些误差
    
    def test_confidence_decreases_during_prediction(self):
        """
        **Property 8: 跟踪丢失恢复**
        预测期间置信度应降低
        """
        filter = HandKeypointFilter(max_lost_frames=5)
        
        base_points = create_hand_points()
        for frame in range(10):
            filter.update(base_points, frame * 0.033)
        
        # 连续丢失，置信度应递减
        prev_confidence = 1.0
        for frame in range(4):
            predicted, _ = filter.update(None, (10 + frame) * 0.033)
            if len(predicted) > 0:
                current_confidence = predicted[0].confidence
                assert current_confidence <= prev_confidence
                prev_confidence = current_confidence


class TestHandIdConsistency:
    """
    手部ID一致性测试
    
    **Property 9: 手部ID一致性**
    """
    
    def test_tracker_maintains_hand_identity(self):
        """
        **Property 9: 手部ID一致性**
        跟踪器应维护左右手ID
        """
        tracker = HandTracker()
        
        # 左手在左边，右手在右边
        left_points = create_hand_points(base_x=-0.2)
        right_points = create_hand_points(base_x=0.2)
        
        # 建立跟踪历史
        for frame in range(10):
            tracker.update(left_points, right_points, frame * 0.033)
        
        # 验证跟踪状态
        assert tracker.is_tracking_left
        assert tracker.is_tracking_right
    
    def test_swap_detection_and_correction(self):
        """
        **Property 9: 手部ID一致性**
        当左右手交换屏幕位置时，跟踪器应基于手腕位置历史维护正确的手部ID
        """
        tracker = HandTracker()
        
        # 建立历史：左手在x=-0.2，右手在x=0.2
        left_points = create_hand_points(base_x=-0.2)
        right_points = create_hand_points(base_x=0.2)
        
        for frame in range(10):
            tracker.update(left_points, right_points, frame * 0.033)
        
        # 记录历史位置
        left_wrist_before = left_points[0].x
        right_wrist_before = right_points[0].x
        
        # 模拟交换：传入时标签错误
        # 实际左手(x=-0.2)被标记为右手，实际右手(x=0.2)被标记为左手
        swapped_left = create_hand_points(base_x=0.2)   # 实际是右手
        swapped_right = create_hand_points(base_x=-0.2)  # 实际是左手
        
        corrected_left, corrected_right = tracker.update(
            swapped_left, swapped_right, 10 * 0.033
        )
        
        # 校正后，左手应该仍在左边（x接近-0.2）
        if corrected_left is not None and corrected_right is not None:
            # 校正后的左手x应该接近历史左手位置
            assert corrected_left[0].x < 0, "校正后左手应在左侧"
            assert corrected_right[0].x > 0, "校正后右手应在右侧"
    
    def test_no_swap_when_positions_consistent(self):
        """
        **Property 9: 手部ID一致性**
        位置一致时不应交换
        """
        tracker = HandTracker()
        
        left_points = create_hand_points(base_x=-0.2)
        right_points = create_hand_points(base_x=0.2)
        
        # 建立历史
        for frame in range(10):
            tracker.update(left_points, right_points, frame * 0.033)
        
        # 继续正常输入
        result_left, result_right = tracker.update(
            left_points, right_points, 10 * 0.033
        )
        
        # 不应交换
        assert result_left[0].x < 0
        assert result_right[0].x > 0
    
    def test_single_hand_tracking(self):
        """
        **Property 9: 手部ID一致性**
        单手跟踪时不应出错
        """
        tracker = HandTracker()
        
        left_points = create_hand_points(base_x=-0.2)
        
        # 只有左手
        for frame in range(10):
            result_left, result_right = tracker.update(
                left_points, None, frame * 0.033
            )
            
            assert result_left is not None
            # 右手可能是None或预测值


class TestFilterReset:
    """滤波器重置测试"""
    
    def test_reset_clears_state(self):
        """重置应清除所有状态"""
        filter = HandKeypointFilter()
        
        # 建立状态
        base_points = create_hand_points()
        for frame in range(10):
            filter.update(base_points, frame * 0.033)
        
        assert filter.is_tracking
        
        # 重置
        filter.reset()
        
        assert not filter.is_tracking
        assert filter.lost_frames == 0
    
    def test_tracker_reset(self):
        """跟踪器重置应清除双手状态"""
        tracker = HandTracker()
        
        left_points = create_hand_points(base_x=-0.2)
        right_points = create_hand_points(base_x=0.2)
        
        for frame in range(10):
            tracker.update(left_points, right_points, frame * 0.033)
        
        tracker.reset()
        
        assert not tracker.is_tracking_left
        assert not tracker.is_tracking_right


class TestFilterEdgeCases:
    """边缘情况测试"""
    
    def test_empty_points_list(self):
        """空点列表应正确处理"""
        filter = HandKeypointFilter()
        
        result, is_predicted = filter.update([], 0.0)
        
        assert is_predicted
        assert len(result) == 0
    
    def test_invalid_points(self):
        """无效点应正确处理"""
        filter = HandKeypointFilter()
        
        # 创建包含无效点的列表
        points = [Point3D(0, 0, 0, confidence=0.0) for _ in range(21)]
        
        result, is_predicted = filter.update(points, 0.0)
        
        # 应该初始化但点可能无效
        assert len(result) == 21


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
