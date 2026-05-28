"""
手势识别器属性测试

**Feature: holistic-hand-integration, Property 12: 手势分类正确性**
**Feature: holistic-hand-integration, Property 13: 手势稳定触发**
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
"""
import sys
from pathlib import Path

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.coordinate_transformer import Point3D
from src.core.hand_skeleton import HandSkeleton3D
from src.core.gesture_recognizer import GestureRecognizer, GestureType, GestureResult


def create_fist_hand() -> list:
    """创建握拳手势的手部关键点"""
    # 所有手指弯曲，指尖靠近MCP
    points = []
    
    # 手腕
    points.append(Point3D(0.0, 0.0, 1.0, confidence=0.9))
    
    # 拇指 - 弯曲
    points.append(Point3D(0.02, 0.01, 1.0, confidence=0.9))   # cmc
    points.append(Point3D(0.025, 0.02, 1.0, confidence=0.9))  # mcp
    points.append(Point3D(0.02, 0.025, 1.0, confidence=0.9))  # ip
    points.append(Point3D(0.015, 0.022, 1.0, confidence=0.9)) # tip (靠近mcp)
    
    # 食指 - 弯曲
    points.append(Point3D(0.015, 0.05, 1.0, confidence=0.9))  # mcp
    points.append(Point3D(0.015, 0.055, 1.0, confidence=0.9)) # pip
    points.append(Point3D(0.015, 0.052, 1.0, confidence=0.9)) # dip
    points.append(Point3D(0.015, 0.048, 1.0, confidence=0.9)) # tip (靠近mcp, <3cm)
    
    # 中指 - 弯曲
    points.append(Point3D(0.0, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(0.0, 0.055, 1.0, confidence=0.9))
    points.append(Point3D(0.0, 0.052, 1.0, confidence=0.9))
    points.append(Point3D(0.0, 0.048, 1.0, confidence=0.9))
    
    # 无名指 - 弯曲
    points.append(Point3D(-0.015, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(-0.015, 0.055, 1.0, confidence=0.9))
    points.append(Point3D(-0.015, 0.052, 1.0, confidence=0.9))
    points.append(Point3D(-0.015, 0.048, 1.0, confidence=0.9))
    
    # 小指 - 弯曲
    points.append(Point3D(-0.03, 0.045, 1.0, confidence=0.9))
    points.append(Point3D(-0.03, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(-0.03, 0.047, 1.0, confidence=0.9))
    points.append(Point3D(-0.03, 0.043, 1.0, confidence=0.9))
    
    return points


def create_open_palm_hand() -> list:
    """创建张开手掌手势的手部关键点"""
    # 所有手指伸展，指尖远离MCP (>6cm)
    # 需要确保 tip_to_mcp 距离 > 6cm = 0.06m
    points = []
    
    # 手腕
    points.append(Point3D(0.0, 0.0, 1.0, confidence=0.9))
    
    # 拇指 - 伸展 (CMC到TIP距离需要>6cm)
    points.append(Point3D(0.03, 0.01, 1.0, confidence=0.9))   # cmc (mcp for thumb)
    points.append(Point3D(0.05, 0.03, 1.0, confidence=0.9))   # mcp
    points.append(Point3D(0.07, 0.05, 1.0, confidence=0.9))   # ip
    points.append(Point3D(0.09, 0.07, 1.0, confidence=0.9))   # tip (距离cmc约8cm)
    
    # 食指 - 伸展 (MCP到TIP距离需要>6cm)
    points.append(Point3D(0.02, 0.05, 1.0, confidence=0.9))   # mcp
    points.append(Point3D(0.02, 0.08, 1.0, confidence=0.9))   # pip
    points.append(Point3D(0.02, 0.10, 1.0, confidence=0.9))   # dip
    points.append(Point3D(0.02, 0.12, 1.0, confidence=0.9))   # tip (距离mcp=7cm)
    
    # 中指 - 伸展
    points.append(Point3D(0.0, 0.05, 1.0, confidence=0.9))    # mcp
    points.append(Point3D(0.0, 0.08, 1.0, confidence=0.9))    # pip
    points.append(Point3D(0.0, 0.11, 1.0, confidence=0.9))    # dip
    points.append(Point3D(0.0, 0.13, 1.0, confidence=0.9))    # tip (距离mcp=8cm)
    
    # 无名指 - 伸展
    points.append(Point3D(-0.02, 0.05, 1.0, confidence=0.9))  # mcp
    points.append(Point3D(-0.02, 0.08, 1.0, confidence=0.9))  # pip
    points.append(Point3D(-0.02, 0.10, 1.0, confidence=0.9))  # dip
    points.append(Point3D(-0.02, 0.12, 1.0, confidence=0.9))  # tip (距离mcp=7cm)
    
    # 小指 - 伸展
    points.append(Point3D(-0.04, 0.045, 1.0, confidence=0.9)) # mcp
    points.append(Point3D(-0.04, 0.075, 1.0, confidence=0.9)) # pip
    points.append(Point3D(-0.04, 0.095, 1.0, confidence=0.9)) # dip
    points.append(Point3D(-0.04, 0.115, 1.0, confidence=0.9)) # tip (距离mcp=7cm)
    
    return points


def create_pointing_hand() -> list:
    """创建食指指向手势的手部关键点"""
    # 食指伸展，其他手指弯曲
    points = []
    
    # 手腕
    points.append(Point3D(0.0, 0.0, 1.0, confidence=0.9))
    
    # 拇指 - 半弯曲
    points.append(Point3D(0.02, 0.01, 1.0, confidence=0.9))
    points.append(Point3D(0.03, 0.02, 1.0, confidence=0.9))
    points.append(Point3D(0.035, 0.025, 1.0, confidence=0.9))
    points.append(Point3D(0.03, 0.03, 1.0, confidence=0.9))
    
    # 食指 - 伸展 (>6cm)
    points.append(Point3D(0.02, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(0.02, 0.08, 1.0, confidence=0.9))
    points.append(Point3D(0.02, 0.10, 1.0, confidence=0.9))
    points.append(Point3D(0.02, 0.12, 1.0, confidence=0.9))
    
    # 中指 - 弯曲 (<3cm)
    points.append(Point3D(0.0, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(0.0, 0.055, 1.0, confidence=0.9))
    points.append(Point3D(0.0, 0.052, 1.0, confidence=0.9))
    points.append(Point3D(0.0, 0.048, 1.0, confidence=0.9))
    
    # 无名指 - 弯曲
    points.append(Point3D(-0.015, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(-0.015, 0.055, 1.0, confidence=0.9))
    points.append(Point3D(-0.015, 0.052, 1.0, confidence=0.9))
    points.append(Point3D(-0.015, 0.048, 1.0, confidence=0.9))
    
    # 小指 - 弯曲
    points.append(Point3D(-0.03, 0.045, 1.0, confidence=0.9))
    points.append(Point3D(-0.03, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(-0.03, 0.047, 1.0, confidence=0.9))
    points.append(Point3D(-0.03, 0.043, 1.0, confidence=0.9))
    
    return points


def create_ok_hand() -> list:
    """创建OK手势的手部关键点"""
    # 拇指和食指接触，其他手指伸展
    points = []
    
    # 手腕
    points.append(Point3D(0.0, 0.0, 1.0, confidence=0.9))
    
    # 拇指 - 弯向食指
    points.append(Point3D(0.02, 0.01, 1.0, confidence=0.9))
    points.append(Point3D(0.03, 0.025, 1.0, confidence=0.9))
    points.append(Point3D(0.035, 0.04, 1.0, confidence=0.9))
    points.append(Point3D(0.03, 0.055, 1.0, confidence=0.9))  # tip接近食指tip
    
    # 食指 - 弯向拇指
    points.append(Point3D(0.02, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(0.025, 0.06, 1.0, confidence=0.9))
    points.append(Point3D(0.03, 0.065, 1.0, confidence=0.9))
    points.append(Point3D(0.032, 0.057, 1.0, confidence=0.9))  # tip接近拇指tip (<2cm)
    
    # 中指 - 伸展
    points.append(Point3D(0.0, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(0.0, 0.08, 1.0, confidence=0.9))
    points.append(Point3D(0.0, 0.10, 1.0, confidence=0.9))
    points.append(Point3D(0.0, 0.12, 1.0, confidence=0.9))
    
    # 无名指 - 伸展
    points.append(Point3D(-0.02, 0.05, 1.0, confidence=0.9))
    points.append(Point3D(-0.02, 0.075, 1.0, confidence=0.9))
    points.append(Point3D(-0.02, 0.095, 1.0, confidence=0.9))
    points.append(Point3D(-0.02, 0.115, 1.0, confidence=0.9))
    
    # 小指 - 伸展
    points.append(Point3D(-0.04, 0.045, 1.0, confidence=0.9))
    points.append(Point3D(-0.04, 0.065, 1.0, confidence=0.9))
    points.append(Point3D(-0.04, 0.085, 1.0, confidence=0.9))
    points.append(Point3D(-0.04, 0.105, 1.0, confidence=0.9))
    
    return points


class TestGestureClassification:
    """
    手势分类正确性测试
    
    **Property 12: 手势分类正确性**
    """
    
    def test_fist_classification(self):
        """
        **Property 12: 手势分类正确性**
        如果所有手指的指尖到MCP距离 < 3cm → 分类为 "fist"
        """
        recognizer = GestureRecognizer()
        points = create_fist_hand()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = recognizer.recognize(skeleton)
        
        assert result.gesture == GestureType.FIST
        assert result.confidence > 0.5
    
    def test_open_palm_classification(self):
        """
        **Property 12: 手势分类正确性**
        如果所有手指的指尖到MCP距离 > 6cm → 分类为 "open_palm"
        """
        recognizer = GestureRecognizer()
        points = create_open_palm_hand()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = recognizer.recognize(skeleton)
        
        assert result.gesture == GestureType.OPEN_PALM
        assert result.confidence > 0.5
    
    def test_pointing_classification(self):
        """
        **Property 12: 手势分类正确性**
        如果仅食指伸展且其他手指弯曲 → 分类为 "pointing"
        """
        recognizer = GestureRecognizer()
        points = create_pointing_hand()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = recognizer.recognize(skeleton)
        
        assert result.gesture == GestureType.POINTING
        assert result.confidence > 0.5
    
    def test_ok_classification(self):
        """
        **Property 12: 手势分类正确性**
        如果拇指尖到食指尖距离 < 2cm 且其他手指伸展 → 分类为 "ok"
        """
        recognizer = GestureRecognizer()
        points = create_ok_hand()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = recognizer.recognize(skeleton)
        
        assert result.gesture == GestureType.OK
        assert result.confidence > 0.5
    
    def test_unknown_for_ambiguous_gesture(self):
        """模糊手势应返回UNKNOWN或低置信度"""
        recognizer = GestureRecognizer()
        
        # 创建模糊手势（部分手指伸展，部分弯曲）
        points = [Point3D(0, i * 0.01, 1.0, confidence=0.9) for i in range(21)]
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        result = recognizer.recognize(skeleton)
        
        # 应该是UNKNOWN或置信度较低
        assert result.gesture == GestureType.UNKNOWN or result.confidence < 0.8


class TestGestureStableTrigger:
    """
    手势稳定触发测试
    
    **Property 13: 手势稳定触发**
    """
    
    def test_trigger_after_stability_threshold(self):
        """
        **Property 13: 手势稳定触发**
        同一手势连续识别达到阈值后应触发
        """
        recognizer = GestureRecognizer(stability_threshold=10, confidence_threshold=0.5)
        points = create_fist_hand()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        triggered = None
        for i in range(15):
            result = recognizer.update(skeleton)
            if result is not None:
                triggered = result
        
        assert triggered == GestureType.FIST
    
    def test_no_trigger_before_threshold(self):
        """
        **Property 13: 手势稳定触发**
        未达到阈值前不应触发
        """
        recognizer = GestureRecognizer(stability_threshold=10, confidence_threshold=0.5)
        points = create_fist_hand()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        triggered_count = 0
        for i in range(9):  # 少于阈值
            result = recognizer.update(skeleton)
            if result is not None:
                triggered_count += 1
        
        assert triggered_count == 0
    
    def test_trigger_only_once_per_gesture(self):
        """
        **Property 13: 手势稳定触发**
        同一手势只应触发一次
        """
        recognizer = GestureRecognizer(stability_threshold=5, confidence_threshold=0.5)
        points = create_fist_hand()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        trigger_count = 0
        for i in range(20):
            result = recognizer.update(skeleton)
            if result is not None:
                trigger_count += 1
        
        assert trigger_count == 1
    
    def test_new_gesture_can_trigger(self):
        """
        **Property 13: 手势稳定触发**
        切换到新手势后可以再次触发
        """
        recognizer = GestureRecognizer(stability_threshold=5, confidence_threshold=0.5)
        
        fist_points = create_fist_hand()
        fist_skeleton = HandSkeleton3D.from_points(fist_points, "left")
        
        open_points = create_open_palm_hand()
        open_skeleton = HandSkeleton3D.from_points(open_points, "left")
        
        triggers = []
        
        # 先触发握拳
        for i in range(10):
            result = recognizer.update(fist_skeleton)
            if result:
                triggers.append(result)
        
        # 再触发张开
        for i in range(10):
            result = recognizer.update(open_skeleton)
            if result:
                triggers.append(result)
        
        assert len(triggers) == 2
        assert GestureType.FIST in triggers
        assert GestureType.OPEN_PALM in triggers
    
    def test_low_confidence_resets_count(self):
        """低置信度应重置稳定计数"""
        recognizer = GestureRecognizer(stability_threshold=5, confidence_threshold=0.8)
        
        fist_points = create_fist_hand()
        fist_skeleton = HandSkeleton3D.from_points(fist_points, "left")
        
        # 低置信度点
        low_conf_points = [Point3D(0, 0, 1.0, confidence=0.3) for _ in range(21)]
        low_conf_skeleton = HandSkeleton3D.from_points(low_conf_points, "left")
        
        # 先累积一些帧
        for i in range(3):
            recognizer.update(fist_skeleton)
        
        # 插入低置信度帧
        recognizer.update(low_conf_skeleton)
        
        # 稳定计数应该重置
        assert recognizer.stable_count == 0


class TestGestureRecognizerState:
    """手势识别器状态测试"""
    
    def test_reset_clears_state(self):
        """reset 应清除所有状态"""
        recognizer = GestureRecognizer(stability_threshold=5)
        points = create_fist_hand()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        # 累积一些状态
        for i in range(10):
            recognizer.update(skeleton)
        
        recognizer.reset()
        
        assert recognizer.current_gesture == GestureType.UNKNOWN
        assert recognizer.stable_count == 0
        assert recognizer.last_triggered is None
    
    def test_current_gesture_property(self):
        """current_gesture 应返回当前识别的手势"""
        recognizer = GestureRecognizer()
        points = create_fist_hand()
        skeleton = HandSkeleton3D.from_points(points, "left")
        
        recognizer.update(skeleton)
        
        assert recognizer.current_gesture == GestureType.FIST


class TestGestureResultSerialization:
    """手势结果序列化测试"""
    
    def test_to_dict(self):
        """to_dict 应返回完整字典"""
        result = GestureResult(
            gesture=GestureType.FIST,
            confidence=0.9,
            hand="left",
            stable_frames=5
        )
        
        data = result.to_dict()
        
        assert data['gesture'] == 'fist'
        assert data['confidence'] == 0.9
        assert data['hand'] == 'left'
        assert data['stable_frames'] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
