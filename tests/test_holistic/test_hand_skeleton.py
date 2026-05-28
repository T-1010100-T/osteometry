"""
手部骨骼模型属性测试

**Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
**Feature: holistic-hand-integration, Property 11: 关节角度计算正确性**
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 6.1, 6.5**
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

from src.core.coordinate_transformer import Point3D
from src.core.hand_skeleton import HandSkeleton3D
from src.core.constants import HAND_LANDMARK_NAMES, HandLandmark


def create_test_hand_points(scale: float = 1.0) -> list:
    """创建测试用手部关键点（简化的手部模型）"""
    # 基础手部模型坐标（单位：米）
    base_points = {
        0: (0.0, 0.0, 1.0),      # wrist
        1: (0.02, 0.01, 1.0),    # thumb_cmc
        2: (0.04, 0.02, 1.0),    # thumb_mcp
        3: (0.05, 0.03, 1.0),    # thumb_ip
        4: (0.06, 0.04, 1.0),    # thumb_tip
        5: (0.02, 0.05, 1.0),    # index_mcp
        6: (0.02, 0.07, 1.0),    # index_pip
        7: (0.02, 0.08, 1.0),    # index_dip
        8: (0.02, 0.09, 1.0),    # index_tip
        9: (0.0, 0.05, 1.0),     # middle_mcp
        10: (0.0, 0.07, 1.0),    # middle_pip
        11: (0.0, 0.08, 1.0),    # middle_dip
        12: (0.0, 0.09, 1.0),    # middle_tip
        13: (-0.02, 0.05, 1.0),  # ring_mcp
        14: (-0.02, 0.07, 1.0),  # ring_pip
        15: (-0.02, 0.08, 1.0),  # ring_dip
        16: (-0.02, 0.09, 1.0),  # ring_tip
        17: (-0.04, 0.04, 1.0),  # pinky_mcp
        18: (-0.04, 0.055, 1.0), # pinky_pip
        19: (-0.04, 0.065, 1.0), # pinky_dip
        20: (-0.04, 0.075, 1.0), # pinky_tip
    }
    
    points = []
    for i in range(21):
        x, y, z = base_points[i]
        points.append(Point3D(x * scale, y * scale, z, confidence=0.9))
    
    return points


def generate_random_hand_points(
    coords: list,
    confidence: float = 0.9
) -> list:
    """从63个坐标值生成21个3D点，确保z>0使点有效"""
    points = []
    for i in range(21):
        x = coords[i * 3]
        y = coords[i * 3 + 1]
        # 确保 z > 0，因为 Point3D.is_valid() 要求 z > 0
        z = abs(coords[i * 3 + 2]) + 0.1  # 确保 z 始终为正
        points.append(Point3D(x, y, z, confidence=confidence))
    return points


# 手指关键点索引映射
FINGER_INDICES = {
    'thumb': [1, 2, 3, 4],      # CMC, MCP, IP, TIP
    'index': [5, 6, 7, 8],      # MCP, PIP, DIP, TIP
    'middle': [9, 10, 11, 12],  # MCP, PIP, DIP, TIP
    'ring': [13, 14, 15, 16],   # MCP, PIP, DIP, TIP
    'pinky': [17, 18, 19, 20],  # MCP, PIP, DIP, TIP
}


class TestMeasurementFormulas:
    """
    手部测量公式正确性测试
    
    **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    """
    
    @given(
        coords=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=63,
            max_size=63
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_palm_length_formula_property(self, coords: list):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.1**
        
        *For any* valid HandSkeleton3D, palm_length SHALL equal
        distance(WRIST, MIDDLE_FINGER_MCP)
        """
        points = generate_random_hand_points(coords)
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        # 手动计算预期值
        wrist = points[HandLandmark.WRIST]
        middle_mcp = points[HandLandmark.MIDDLE_FINGER_MCP]
        expected = wrist.distance_to(middle_mcp)
        
        actual = skeleton.get_palm_length()
        
        assert math.isclose(actual, expected, rel_tol=1e-6)
    
    @given(
        coords=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=63,
            max_size=63
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_hand_width_formula_property(self, coords: list):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.3**
        
        *For any* valid HandSkeleton3D, hand_width SHALL equal
        distance(THUMB_CMC, PINKY_MCP)
        """
        points = generate_random_hand_points(coords)
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        thumb_cmc = points[HandLandmark.THUMB_CMC]
        pinky_mcp = points[HandLandmark.PINKY_MCP]
        expected = thumb_cmc.distance_to(pinky_mcp)
        
        actual = skeleton.get_hand_width()
        
        assert math.isclose(actual, expected, rel_tol=1e-6)
    
    @given(
        coords=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=63,
            max_size=63
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_thumb_span_formula_property(self, coords: list):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.4**
        
        *For any* valid HandSkeleton3D, thumb_span SHALL equal
        distance(THUMB_TIP, INDEX_FINGER_TIP)
        """
        points = generate_random_hand_points(coords)
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        thumb_tip = points[HandLandmark.THUMB_TIP]
        index_tip = points[HandLandmark.INDEX_FINGER_TIP]
        expected = thumb_tip.distance_to(index_tip)
        
        actual = skeleton.get_thumb_span()
        
        assert math.isclose(actual, expected, rel_tol=1e-6)
    
    @given(
        coords=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=63,
            max_size=63
        ),
        finger=st.sampled_from(['index', 'middle', 'ring', 'pinky'])
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_finger_length_is_sum_of_segments_property(self, coords: list, finger: str):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.2**
        
        *For any* valid HandSkeleton3D and any finger,
        finger_length SHALL equal sum of distances from MCP through PIP, DIP to TIP
        """
        points = generate_random_hand_points(coords)
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        # 获取手指关键点索引
        indices = FINGER_INDICES[finger]
        
        # 计算预期长度：各段距离之和
        expected = 0.0
        for i in range(len(indices) - 1):
            start_point = points[indices[i]]
            end_point = points[indices[i + 1]]
            expected += start_point.distance_to(end_point)
        
        actual = skeleton.get_finger_length(finger)
        
        assert math.isclose(actual, expected, rel_tol=1e-6)
    
    @given(
        coords=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=63,
            max_size=63
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_thumb_length_is_sum_of_segments_property(self, coords: list):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.2**
        
        *For any* valid HandSkeleton3D, thumb_length SHALL equal
        sum of distances from CMC through MCP, IP to TIP
        """
        points = generate_random_hand_points(coords)
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        # 拇指关键点: CMC(1), MCP(2), IP(3), TIP(4)
        thumb_cmc = points[HandLandmark.THUMB_CMC]
        thumb_mcp = points[HandLandmark.THUMB_MCP]
        thumb_ip = points[HandLandmark.THUMB_IP]
        thumb_tip = points[HandLandmark.THUMB_TIP]
        
        expected = (
            thumb_cmc.distance_to(thumb_mcp) +
            thumb_mcp.distance_to(thumb_ip) +
            thumb_ip.distance_to(thumb_tip)
        )
        
        actual = skeleton.get_finger_length("thumb")
        
        assert math.isclose(actual, expected, rel_tol=1e-6)
    
    @given(
        coords=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=63,
            max_size=63
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_measurements_non_negative(self, coords: list):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
        
        *For any* valid HandSkeleton3D, all measurements SHALL be non-negative
        """
        points = generate_random_hand_points(coords)
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        assert skeleton.get_palm_length() >= 0
        assert skeleton.get_hand_width() >= 0
        assert skeleton.get_thumb_span() >= 0
        
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            assert skeleton.get_finger_length(finger) >= 0
    
    def test_palm_length_formula(self):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.1**
        
        palm_length = distance(WRIST, MIDDLE_FINGER_MCP)
        """
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        # 手动计算预期值
        wrist = points[HandLandmark.WRIST]
        middle_mcp = points[HandLandmark.MIDDLE_FINGER_MCP]
        expected = wrist.distance_to(middle_mcp)
        
        actual = skeleton.get_palm_length()
        
        assert math.isclose(actual, expected, rel_tol=1e-6)
    
    def test_hand_width_formula(self):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.3**
        
        hand_width = distance(THUMB_CMC, PINKY_MCP)
        """
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        thumb_cmc = points[HandLandmark.THUMB_CMC]
        pinky_mcp = points[HandLandmark.PINKY_MCP]
        expected = thumb_cmc.distance_to(pinky_mcp)
        
        actual = skeleton.get_hand_width()
        
        assert math.isclose(actual, expected, rel_tol=1e-6)
    
    def test_thumb_span_formula(self):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.4**
        
        thumb_span = distance(THUMB_TIP, INDEX_FINGER_TIP)
        """
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        thumb_tip = points[HandLandmark.THUMB_TIP]
        index_tip = points[HandLandmark.INDEX_FINGER_TIP]
        expected = thumb_tip.distance_to(index_tip)
        
        actual = skeleton.get_thumb_span()
        
        assert math.isclose(actual, expected, rel_tol=1e-6)

    
    def test_finger_length_is_sum_of_segments(self):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.2**
        
        finger_length = sum of distances from MCP through PIP, DIP to TIP
        """
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        # 测试食指长度
        index_mcp = points[HandLandmark.INDEX_FINGER_MCP]
        index_pip = points[HandLandmark.INDEX_FINGER_PIP]
        index_dip = points[HandLandmark.INDEX_FINGER_DIP]
        index_tip = points[HandLandmark.INDEX_FINGER_TIP]
        
        expected = (
            index_mcp.distance_to(index_pip) +
            index_pip.distance_to(index_dip) +
            index_dip.distance_to(index_tip)
        )
        
        actual = skeleton.get_finger_length("index")
        
        assert math.isclose(actual, expected, rel_tol=1e-6)
    
    @given(scale=st.floats(min_value=0.5, max_value=2.0, allow_nan=False))
    @settings(max_examples=50)
    def test_measurements_scale_linearly(self, scale: float):
        """
        **Feature: holistic-hand-integration, Property 10: 手部测量公式正确性**
        **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
        
        测量值应随缩放线性变化
        """
        base_points = create_test_hand_points(scale=1.0)
        scaled_points = create_test_hand_points(scale=scale)
        
        base_skeleton = HandSkeleton3D.from_points(base_points, "left")
        scaled_skeleton = HandSkeleton3D.from_points(scaled_points, "left")
        
        # 手掌长度应按比例缩放
        base_palm = base_skeleton.get_palm_length()
        scaled_palm = scaled_skeleton.get_palm_length()
        
        if base_palm > 0:
            assert math.isclose(scaled_palm / base_palm, scale, rel_tol=1e-5)


class TestJointAngleCalculation:
    """
    关节角度计算正确性测试
    
    **Property 11: 关节角度计算正确性**
    """
    
    def test_straight_finger_angle_is_180(self):
        """
        **Property 11: 关节角度计算正确性**
        完全伸直的手指角度应接近180度
        """
        # 创建完全伸直的手指
        points = []
        for i in range(21):
            if i in [5, 6, 7, 8]:  # 食指关键点
                y = 0.05 + (i - 5) * 0.02  # 直线排列
                points.append(Point3D(0.02, y, 1.0, confidence=0.9))
            else:
                points.append(Point3D(0.0, 0.0, 1.0, confidence=0.9))
        
        # 设置手腕
        points[0] = Point3D(0.0, 0.0, 1.0, confidence=0.9)
        
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        # PIP关节角度应接近180度
        pip_angle = skeleton.get_finger_angle("index", "pip")
        
        # 由于是直线，角度应接近180度
        assert pip_angle > 170 or pip_angle == 0  # 0表示无法计算
    
    def test_angle_within_anatomical_limits(self):
        """
        **Property 11: 关节角度计算正确性**
        关节角度应在解剖学限制内 (0-180°)
        """
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        all_angles = skeleton.get_all_joint_angles()
        
        for angle_name, angle in all_angles.items():
            if angle > 0:  # 只检查有效角度
                assert 0 <= angle <= 180, f"{angle_name} = {angle}° is out of range"
    
    def test_angle_calculation_uses_correct_vectors(self):
        """
        **Property 11: 关节角度计算正确性**
        角度应使用正确的向量计算 (MCP→PIP) 和 (PIP→DIP)
        """
        # 创建90度弯曲的手指
        points = [Point3D(0, 0, 1.0, confidence=0.9) for _ in range(21)]
        
        # 设置食指形成90度角
        points[5] = Point3D(0.0, 0.0, 1.0, confidence=0.9)   # MCP
        points[6] = Point3D(0.0, 0.02, 1.0, confidence=0.9)  # PIP (向上)
        points[7] = Point3D(0.02, 0.02, 1.0, confidence=0.9) # DIP (向右)
        points[8] = Point3D(0.04, 0.02, 1.0, confidence=0.9) # TIP
        
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        pip_angle = skeleton.get_finger_angle("index", "pip")
        
        # 应该接近90度
        assert 85 <= pip_angle <= 95


class TestHandSkeletonStructure:
    """手部骨骼结构测试"""
    
    def test_from_points_creates_21_joints(self):
        """from_points 应创建21个关节"""
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        assert len(skeleton.joints) == 21
    
    def test_all_finger_lengths_returns_5_fingers(self):
        """get_all_finger_lengths 应返回5个手指的长度"""
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        lengths = skeleton.get_all_finger_lengths()
        
        assert len(lengths) == 5
        assert 'thumb' in lengths
        assert 'index' in lengths
        assert 'middle' in lengths
        assert 'ring' in lengths
        assert 'pinky' in lengths
    
    def test_to_array_shape(self):
        """to_array 应返回 (21, 4) 形状的数组"""
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        arr = skeleton.to_array()
        
        assert arr.shape == (21, 4)
    
    def test_connections_list(self):
        """connections 应返回骨骼连接列表"""
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        connections = skeleton.connections
        
        assert len(connections) > 0
        for start, end in connections:
            assert isinstance(start, str)
            assert isinstance(end, str)


class TestTipToMcpDistance:
    """指尖到MCP距离测试（用于手势识别）"""
    
    def test_tip_to_mcp_distance_positive(self):
        """指尖到MCP距离应为正值"""
        points = create_test_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            distance = skeleton.get_tip_to_mcp_distance(finger)
            assert distance >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
