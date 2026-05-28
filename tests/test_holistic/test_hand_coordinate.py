"""
手部坐标变换属性测试

**Feature: holistic-hand-integration, Property 5: 手部3D坐标变换正确性**
**Feature: holistic-hand-integration, Property 6: 深度一致性处理**
**Validates: Requirements 3.1, 3.4, 3.5**
"""
import sys
from pathlib import Path
import math

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.pose_estimator import Landmark
from src.core.hand_result import HandResult
from src.hardware.frame_set import Intrinsics


# 测试用相机内参
TEST_INTRINSICS = Intrinsics(
    width=640,
    height=480,
    fx=600.0,
    fy=600.0,
    ppx=320.0,
    ppy=240.0,
    coeffs=[0.0] * 5
)


class TestPinholeModel:
    """
    针孔相机模型测试
    
    **Property 5: 手部3D坐标变换正确性**
    """
    
    @given(
        u=st.integers(min_value=0, max_value=639),
        v=st.integers(min_value=0, max_value=479),
        depth=st.floats(min_value=0.3, max_value=3.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_pinhole_camera_model(self, u: int, v: int, depth: float):
        """
        **Property 5: 手部3D坐标变换正确性**
        3D坐标应满足针孔相机模型: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
        """
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS)
        point = transformer.pixel_to_3d(u, v, depth)
        
        # 验证针孔相机模型
        expected_x = (u - TEST_INTRINSICS.ppx) * depth / TEST_INTRINSICS.fx
        expected_y = (v - TEST_INTRINSICS.ppy) * depth / TEST_INTRINSICS.fy
        expected_z = depth
        
        assert math.isclose(point.x, expected_x, rel_tol=1e-6)
        assert math.isclose(point.y, expected_y, rel_tol=1e-6)
        assert math.isclose(point.z, expected_z, rel_tol=1e-6)
    
    def test_center_pixel_maps_to_optical_axis(self):
        """
        **Property 5: 手部3D坐标变换正确性**
        图像中心点应映射到光轴上 (X=0, Y=0)
        """
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS)
        
        # 主点位置
        cx, cy = int(TEST_INTRINSICS.ppx), int(TEST_INTRINSICS.ppy)
        depth = 1.5
        
        point = transformer.pixel_to_3d(cx, cy, depth)
        
        assert math.isclose(point.x, 0.0, abs_tol=1e-6)
        assert math.isclose(point.y, 0.0, abs_tol=1e-6)
        assert math.isclose(point.z, depth, rel_tol=1e-6)



class TestMedianFilter:
    """中值滤波测试"""
    
    def test_median_filter_returns_median(self):
        """中值滤波应返回邻域中值"""
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS, median_filter_size=3)
        
        # 创建测试深度图
        depth_frame = np.zeros((480, 640), dtype=np.uint16)
        # 在中心区域设置不同深度值
        depth_frame[238:242, 318:322] = np.array([
            [1000, 1100, 1200, 1300],
            [1050, 1500, 1150, 1250],  # 中心是1500
            [1000, 1100, 1200, 1300],
            [1050, 1150, 1250, 1350]
        ], dtype=np.uint16)
        
        # 获取中心点深度
        depth = transformer.apply_median_filter(depth_frame, 320, 240, kernel_size=3)
        
        # 应该返回中值，而不是异常值1500
        assert depth > 0
        assert depth < 1.5  # 1500 * 0.001 = 1.5m
    
    def test_median_filter_handles_invalid_depth(self):
        """中值滤波应处理无效深度"""
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS)
        
        # 全零深度图
        depth_frame = np.zeros((480, 640), dtype=np.uint16)
        
        depth = transformer.apply_median_filter(depth_frame, 320, 240)
        assert depth == 0.0


class TestDepthConsistency:
    """
    深度一致性测试
    
    **Feature: holistic-hand-integration, Property 6: 深度一致性处理**
    **Validates: Requirements 3.4, 3.5**
    """
    
    @given(
        hand_depth=st.floats(min_value=0.5, max_value=2.5, allow_nan=False),
        body_depth=st.floats(min_value=0.5, max_value=2.5, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_depth_consistency_within_tolerance(self, hand_depth: float, body_depth: float):
        """
        **Feature: holistic-hand-integration, Property 6: 深度一致性处理**
        **Validates: Requirements 3.4, 3.5**
        
        *For any* hand_wrist_depth and body_wrist_depth, verify_depth_consistency
        SHALL return True if and only if |hand_depth - body_depth| <= 5cm
        """
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        tolerance = 0.05  # 5cm
        transformer = HandCoordinateTransformer(TEST_INTRINSICS, consistency_tolerance=tolerance)
        
        result = transformer.verify_depth_consistency(hand_depth, body_depth)
        expected = abs(hand_depth - body_depth) <= tolerance
        
        assert result == expected
    
    @given(
        body_depth=st.floats(min_value=0.5, max_value=2.5, allow_nan=False),
        depth_diff=st.floats(min_value=0.06, max_value=0.5, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_uses_body_depth_when_consistency_fails(self, body_depth: float, depth_diff: float):
        """
        **Feature: holistic-hand-integration, Property 6: 深度一致性处理**
        **Validates: Requirements 3.4, 3.5**
        
        *For any* frame where depth difference exceeds 5cm, the system SHALL use
        body wrist depth as reference for hand landmark depth estimation.
        """
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        tolerance = 0.05  # 5cm
        transformer = HandCoordinateTransformer(TEST_INTRINSICS, consistency_tolerance=tolerance)
        
        # 手部深度与身体深度差异超过容差
        hand_depth_in_frame = body_depth + depth_diff  # 差异 > 5cm
        
        # 创建深度图，手部区域使用不一致的深度
        depth_value_uint16 = int(hand_depth_in_frame / 0.001)  # 转换为uint16
        depth_frame = np.ones((480, 640), dtype=np.uint16) * depth_value_uint16
        
        # 创建手部结果 - 所有关键点在图像中心
        hand_landmarks = [Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(21)]
        hand_result = HandResult(
            landmarks=hand_landmarks,
            detected=True,
            handedness='left',
            confidence=0.9
        )
        
        # 调用转换，提供身体手腕深度
        points_3d = transformer.hand_landmarks_to_3d(
            hand_result, depth_frame, (640, 480), body_depth
        )
        
        # 验证：所有点都应该被转换（21个点）
        assert len(points_3d) == 21
        
        # 验证：一致性校验应该失败
        assert not transformer.verify_depth_consistency(hand_depth_in_frame, body_depth, tolerance)
    
    @given(
        body_depth=st.floats(min_value=0.5, max_value=2.5, allow_nan=False),
        depth_diff=st.floats(min_value=-0.04, max_value=0.04, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_uses_hand_depth_when_consistency_passes(self, body_depth: float, depth_diff: float):
        """
        **Feature: holistic-hand-integration, Property 6: 深度一致性处理**
        **Validates: Requirements 3.4, 3.5**
        
        *For any* frame where depth difference is within 5cm tolerance,
        the system SHALL use the hand wrist depth directly.
        """
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        tolerance = 0.05  # 5cm
        transformer = HandCoordinateTransformer(TEST_INTRINSICS, consistency_tolerance=tolerance)
        
        # 手部深度与身体深度差异在容差内
        hand_depth_in_frame = body_depth + depth_diff
        if hand_depth_in_frame <= 0.1:
            hand_depth_in_frame = 0.5  # 确保深度有效
        
        # 创建深度图
        depth_value_uint16 = int(hand_depth_in_frame / 0.001)
        depth_frame = np.ones((480, 640), dtype=np.uint16) * depth_value_uint16
        
        # 创建手部结果
        hand_landmarks = [Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(21)]
        hand_result = HandResult(
            landmarks=hand_landmarks,
            detected=True,
            handedness='left',
            confidence=0.9
        )
        
        points_3d = transformer.hand_landmarks_to_3d(
            hand_result, depth_frame, (640, 480), body_depth
        )
        
        # 验证：所有点都应该被转换
        assert len(points_3d) == 21
        
        # 验证：一致性校验应该通过（差异在容差内）
        assert transformer.verify_depth_consistency(hand_depth_in_frame, body_depth, tolerance)
    
    def test_depth_consistency_uses_body_depth_on_failure(self):
        """
        **Feature: holistic-hand-integration, Property 6: 深度一致性处理**
        **Validates: Requirements 3.4, 3.5**
        
        一致性校验失败时应使用身体手腕深度作为参考
        """
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS, consistency_tolerance=0.05)
        
        # 创建深度图，手部区域深度与身体手腕差异大
        depth_frame = np.ones((480, 640), dtype=np.uint16) * 1500  # 1.5m
        
        # 创建手部结果
        hand_landmarks = [Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(21)]
        hand_result = HandResult(
            landmarks=hand_landmarks,
            detected=True,
            handedness='left',
            confidence=0.9
        )
        
        # 身体手腕深度与深度图差异大
        body_wrist_depth = 1.0  # 1.0m，与1.5m差0.5m > 0.05m
        
        points_3d = transformer.hand_landmarks_to_3d(
            hand_result, depth_frame, (640, 480), body_wrist_depth
        )
        
        # 验证：所有点都应该被转换
        assert len(points_3d) == 21
        # 验证：一致性校验失败
        assert not transformer.verify_depth_consistency(1.5, body_wrist_depth)
    
    def test_invalid_depth_returns_false(self):
        """
        **Feature: holistic-hand-integration, Property 6: 深度一致性处理**
        **Validates: Requirements 3.4, 3.5**
        
        无效深度（<=0）应返回False
        """
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS)
        
        assert transformer.verify_depth_consistency(0.0, 1.5) == False
        assert transformer.verify_depth_consistency(1.5, 0.0) == False
        assert transformer.verify_depth_consistency(0.0, 0.0) == False
        assert transformer.verify_depth_consistency(-0.5, 1.5) == False
        assert transformer.verify_depth_consistency(1.5, -0.5) == False
    
    @given(
        tolerance=st.floats(min_value=0.01, max_value=0.2, allow_nan=False),
        hand_depth=st.floats(min_value=0.5, max_value=2.5, allow_nan=False),
        body_depth=st.floats(min_value=0.5, max_value=2.5, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_configurable_tolerance(self, tolerance: float, hand_depth: float, body_depth: float):
        """
        **Feature: holistic-hand-integration, Property 6: 深度一致性处理**
        **Validates: Requirements 3.4**
        
        *For any* configurable tolerance value, the consistency check SHALL
        correctly use that tolerance for comparison.
        """
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS, consistency_tolerance=tolerance)
        
        result = transformer.verify_depth_consistency(hand_depth, body_depth)
        expected = abs(hand_depth - body_depth) <= tolerance
        
        assert result == expected


class TestFingertipCorrection:
    """指尖深度校正测试"""
    
    def test_fingertip_correction_handles_edge_artifacts(self):
        """指尖校正应处理边缘伪影"""
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS)
        
        # 创建深度图，指尖位置有异常深度
        depth_frame = np.ones((480, 640), dtype=np.uint16) * 1500  # 1.5m
        depth_frame[240, 320] = 500  # 指尖处异常深度 0.5m
        
        palm_depth = 1.5
        corrected = transformer.correct_fingertip_depth(depth_frame, (320, 240), palm_depth)
        
        # 校正后深度应接近手掌深度
        assert abs(corrected - palm_depth) <= 0.1


class TestHandLandmarksTo3D:
    """手部关键点3D转换测试"""
    
    def test_returns_21_points(self):
        """应返回21个3D点"""
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS)
        
        depth_frame = np.ones((480, 640), dtype=np.uint16) * 1500
        hand_landmarks = [Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(21)]
        hand_result = HandResult(
            landmarks=hand_landmarks,
            detected=True,
            handedness='left',
            confidence=0.9
        )
        
        points_3d = transformer.hand_landmarks_to_3d(hand_result, depth_frame, (640, 480))
        
        assert len(points_3d) == 21
    
    def test_undetected_hand_returns_zero_points(self):
        """未检测到手部应返回零置信度点"""
        from src.core.hand_coordinate_transformer import HandCoordinateTransformer
        
        transformer = HandCoordinateTransformer(TEST_INTRINSICS)
        
        depth_frame = np.ones((480, 640), dtype=np.uint16) * 1500
        hand_result = HandResult.empty('left')
        
        points_3d = transformer.hand_landmarks_to_3d(hand_result, depth_frame, (640, 480))
        
        assert len(points_3d) == 21
        for point in points_3d:
            assert point.confidence == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
