"""
手部测量引擎测试

**Feature: holistic-hand-integration**
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4**
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
from src.measurement.hand_measurement import HandMeasurementEngine, HandMeasurementResult


def create_realistic_hand_points(scale: float = 1.0) -> list:
    """
    创建逼真的手部关键点
    
    基于成人手部平均尺寸（手掌长约9.5cm，手宽约8.5cm）
    """
    # 基础手部模型坐标（单位：米）
    # 手腕在原点，手指向Y正方向延伸
    base_points = {
        0: (0.0, 0.0, 1.0),           # wrist
        1: (0.035, 0.01, 1.0),        # thumb_cmc
        2: (0.05, 0.025, 1.0),        # thumb_mcp
        3: (0.055, 0.04, 1.0),        # thumb_ip
        4: (0.06, 0.055, 1.0),        # thumb_tip
        5: (0.025, 0.095, 1.0),       # index_mcp
        6: (0.025, 0.125, 1.0),       # index_pip
        7: (0.025, 0.145, 1.0),       # index_dip
        8: (0.025, 0.165, 1.0),       # index_tip
        9: (0.0, 0.095, 1.0),         # middle_mcp (手掌长度参考点)
        10: (0.0, 0.13, 1.0),         # middle_pip
        11: (0.0, 0.155, 1.0),        # middle_dip
        12: (0.0, 0.18, 1.0),         # middle_tip
        13: (-0.025, 0.09, 1.0),      # ring_mcp
        14: (-0.025, 0.12, 1.0),      # ring_pip
        15: (-0.025, 0.14, 1.0),      # ring_dip
        16: (-0.025, 0.16, 1.0),      # ring_tip
        17: (-0.05, 0.08, 1.0),       # pinky_mcp (手宽参考点)
        18: (-0.05, 0.10, 1.0),       # pinky_pip
        19: (-0.05, 0.115, 1.0),      # pinky_dip
        20: (-0.05, 0.13, 1.0),       # pinky_tip
    }
    
    points = []
    for i in range(21):
        x, y, z = base_points[i]
        points.append(Point3D(x * scale, y * scale, z, confidence=0.9))
    
    return points


class TestHandMeasurementEngine:
    """手部测量引擎测试"""
    
    def test_calculate_all_returns_valid_result(self):
        """calculate_all 应返回有效的测量结果"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        
        assert isinstance(result, HandMeasurementResult)
        assert result.handedness == "left"
        assert result.is_valid
    
    def test_palm_length_in_realistic_range(self):
        """手掌长度应在合理范围内 (5-15cm)"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        
        # 手掌长度约9.5cm
        assert 0.05 <= result.palm_length <= 0.15
        assert 0.08 <= result.palm_length <= 0.11  # 更精确的范围
    
    def test_hand_width_in_realistic_range(self):
        """手宽应在合理范围内 (5-12cm)"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        
        # 手宽约8.5cm
        assert 0.05 <= result.hand_width <= 0.12
    
    def test_finger_lengths_all_present(self):
        """应返回所有5个手指的长度"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        
        assert len(result.finger_lengths) == 5
        assert 'thumb' in result.finger_lengths
        assert 'index' in result.finger_lengths
        assert 'middle' in result.finger_lengths
        assert 'ring' in result.finger_lengths
        assert 'pinky' in result.finger_lengths
    
    def test_finger_lengths_positive(self):
        """手指长度应为正值"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        
        for finger, length in result.finger_lengths.items():
            assert length > 0, f"{finger} 长度应为正值"
    
    def test_palm_circumference_estimated(self):
        """应估算掌围"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        
        # 掌围通常在15-25cm之间
        assert result.palm_circumference > 0
    
    def test_joint_angles_calculated(self):
        """应计算关节角度"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        
        assert len(result.joint_angles) > 0


class TestMeasurementResultSerialization:
    """测量结果序列化测试"""
    
    def test_to_dict(self):
        """to_dict 应返回完整字典"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        data = result.to_dict()
        
        assert 'palm_length' in data
        assert 'hand_width' in data
        assert 'finger_lengths' in data
        assert 'handedness' in data
    
    def test_to_json(self):
        """to_json 应返回有效JSON"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        json_str = result.to_json()
        
        import json
        parsed = json.loads(json_str)
        assert parsed['handedness'] == 'left'
    
    def test_get_linear_measurements_mm(self):
        """get_linear_measurements_mm 应返回毫米单位"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        mm_data = result.get_linear_measurements_mm()
        
        # 手掌长度约95mm
        assert 50 <= mm_data['手掌长度'] <= 150


class TestMeasurementHistory:
    """测量历史测试"""
    
    def test_history_accumulates(self):
        """历史应累积"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        for i in range(5):
            engine.calculate_all(skeleton, timestamp=i * 0.033)
        
        assert engine.history_count['left'] == 5
    
    def test_get_averaged_result(self):
        """应返回平均结果"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        for i in range(5):
            engine.calculate_all(skeleton, timestamp=i * 0.033)
        
        avg = engine.get_averaged_result("left")
        
        assert avg is not None
        assert avg.is_valid
        assert avg.palm_length > 0
    
    def test_clear_history(self):
        """clear_history 应清空历史"""
        engine = HandMeasurementEngine()
        points = create_realistic_hand_points()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        for i in range(5):
            engine.calculate_all(skeleton)
        
        engine.clear_history("left")
        
        assert engine.history_count['left'] == 0


class TestLowConfidenceHandling:
    """低置信度处理测试"""
    
    def test_low_confidence_returns_invalid(self):
        """低置信度应返回无效结果"""
        engine = HandMeasurementEngine(min_confidence=0.8)
        
        # 创建低置信度点
        points = [Point3D(0, 0, 1.0, confidence=0.3) for _ in range(21)]
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = engine.calculate_all(skeleton)
        
        assert not result.is_valid


class TestMeasurementScaling:
    """测量缩放测试"""
    
    @given(scale=st.floats(min_value=0.8, max_value=1.2, allow_nan=False))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_measurements_scale_with_hand_size(self, scale: float):
        """测量值应随手部大小缩放"""
        engine = HandMeasurementEngine()
        
        base_points = create_realistic_hand_points(scale=1.0)
        scaled_points = create_realistic_hand_points(scale=scale)
        
        base_skeleton = HandSkeleton3D.from_points(base_points, "left")
        scaled_skeleton = HandSkeleton3D.from_points(scaled_points, "left")
        
        base_result = engine.calculate_all(base_skeleton)
        engine.clear_history()
        scaled_result = engine.calculate_all(scaled_skeleton)
        
        if base_result.palm_length > 0 and scaled_result.palm_length > 0:
            ratio = scaled_result.palm_length / base_result.palm_length
            assert math.isclose(ratio, scale, rel_tol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
