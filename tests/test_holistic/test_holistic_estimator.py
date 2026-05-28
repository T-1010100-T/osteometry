"""
Holistic 估计器属性测试

**Feature: holistic-hand-integration, Property 1: Holistic 模型输出结构正确性**
**Feature: holistic-hand-integration, Property 2: API 向后兼容性**
**Validates: Requirements 1.2, 1.3, 1.4, 2.1**
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.pose_estimator import Landmark, PoseResult
from src.core.hand_result import HandResult, HolisticResult
from src.core.constants import BODY_LANDMARK_COUNT, HAND_LANDMARK_COUNT


class MockLandmark:
    """模拟 MediaPipe Landmark"""
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MockLandmarkList:
    """模拟 MediaPipe LandmarkList"""
    def __init__(self, count):
        self.landmark = [MockLandmark() for _ in range(count)]


class MockHolisticResults:
    """模拟 MediaPipe Holistic 结果"""
    def __init__(self, pose=True, left_hand=False, right_hand=False):
        self.pose_landmarks = MockLandmarkList(33) if pose else None
        self.pose_world_landmarks = MockLandmarkList(33) if pose else None
        self.left_hand_landmarks = MockLandmarkList(21) if left_hand else None
        self.right_hand_landmarks = MockLandmarkList(21) if right_hand else None


class TestHolisticOutputStructure:
    """
    Holistic 模型输出结构正确性测试
    
    **Property 1: Holistic 模型输出结构正确性**
    """
    
    def test_body_landmarks_count(self):
        """
        **Property 1: Holistic 模型输出结构正确性**
        身体关键点应有33个
        """
        from src.core.holistic_estimator import HolisticEstimator
        
        with patch.object(HolisticEstimator, '__init__', lambda self, **kwargs: None):
            estimator = HolisticEstimator()
            estimator._holistic = Mock()
            estimator._holistic.process = Mock(return_value=MockHolisticResults(pose=True))
            estimator._frame_count = 0
            estimator._detect_count = 0
            estimator._hand_detection_interval = 1
            estimator._last_hand_detection_frame = 0
            estimator._last_latency = 0
            estimator._adaptive_interval = False
            estimator._fallback_mode = False
            estimator._consecutive_hand_errors = 0
            estimator._max_hand_errors = 10
            estimator.enable_hands = True
            
            # 需要导入实际方法
            from src.core.holistic_estimator import HolisticEstimator as RealEstimator
            estimator._parse_pose_landmarks = RealEstimator._parse_pose_landmarks.__get__(estimator)
            estimator._parse_hand_landmarks = RealEstimator._parse_hand_landmarks.__get__(estimator)
            estimator._should_detect_hands = RealEstimator._should_detect_hands.__get__(estimator)
            estimator._adjust_detection_interval = RealEstimator._adjust_detection_interval.__get__(estimator)
            estimator.detect = RealEstimator.detect.__get__(estimator)
            
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = estimator.detect(image)
            
            assert result.pose.detected == True
            assert len(result.pose.landmarks) == BODY_LANDMARK_COUNT
    
    def test_hand_landmarks_count_when_detected(self):
        """
        **Property 1: Holistic 模型输出结构正确性**
        检测到手部时应有21个关键点
        """
        from src.core.holistic_estimator import HolisticEstimator
        
        with patch.object(HolisticEstimator, '__init__', lambda self, **kwargs: None):
            estimator = HolisticEstimator()
            estimator._holistic = Mock()
            estimator._holistic.process = Mock(return_value=MockHolisticResults(
                pose=True, left_hand=True, right_hand=True
            ))
            estimator._frame_count = 0
            estimator._detect_count = 0
            estimator._hand_detection_interval = 1
            estimator._last_hand_detection_frame = 0
            estimator._last_latency = 0
            estimator._adaptive_interval = False
            estimator._fallback_mode = False
            estimator._consecutive_hand_errors = 0
            estimator._max_hand_errors = 10
            estimator.enable_hands = True
            
            from src.core.holistic_estimator import HolisticEstimator as RealEstimator
            estimator._parse_pose_landmarks = RealEstimator._parse_pose_landmarks.__get__(estimator)
            estimator._parse_hand_landmarks = RealEstimator._parse_hand_landmarks.__get__(estimator)
            estimator._should_detect_hands = RealEstimator._should_detect_hands.__get__(estimator)
            estimator._adjust_detection_interval = RealEstimator._adjust_detection_interval.__get__(estimator)
            estimator.detect = RealEstimator.detect.__get__(estimator)
            
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = estimator.detect(image)
            
            assert result.left_hand.detected == True
            assert len(result.left_hand.landmarks) == HAND_LANDMARK_COUNT
            assert result.right_hand.detected == True
            assert len(result.right_hand.landmarks) == HAND_LANDMARK_COUNT

    
    def test_hand_not_detected_returns_empty(self):
        """
        **Property 1: Holistic 模型输出结构正确性**
        未检测到手部时应返回空结果
        """
        from src.core.holistic_estimator import HolisticEstimator
        
        with patch.object(HolisticEstimator, '__init__', lambda self, **kwargs: None):
            estimator = HolisticEstimator()
            estimator._holistic = Mock()
            estimator._holistic.process = Mock(return_value=MockHolisticResults(
                pose=True, left_hand=False, right_hand=False
            ))
            estimator._frame_count = 0
            estimator._detect_count = 0
            estimator._hand_detection_interval = 1
            estimator._last_hand_detection_frame = 0
            estimator._last_latency = 0
            estimator._adaptive_interval = False
            estimator._fallback_mode = False
            estimator._consecutive_hand_errors = 0
            estimator._max_hand_errors = 10
            estimator.enable_hands = True
            
            from src.core.holistic_estimator import HolisticEstimator as RealEstimator
            estimator._parse_pose_landmarks = RealEstimator._parse_pose_landmarks.__get__(estimator)
            estimator._parse_hand_landmarks = RealEstimator._parse_hand_landmarks.__get__(estimator)
            estimator._should_detect_hands = RealEstimator._should_detect_hands.__get__(estimator)
            estimator._adjust_detection_interval = RealEstimator._adjust_detection_interval.__get__(estimator)
            estimator.detect = RealEstimator.detect.__get__(estimator)
            
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = estimator.detect(image)
            
            assert result.left_hand.detected == False
            assert len(result.left_hand.landmarks) == 0
            assert result.right_hand.detected == False
            assert len(result.right_hand.landmarks) == 0


class TestAPIBackwardCompatibility:
    """
    API 向后兼容性测试
    
    **Property 2: API 向后兼容性**
    """
    
    def test_detect_pose_only_returns_pose_result(self):
        """
        **Property 2: API 向后兼容性**
        detect_pose_only 应返回 PoseResult 类型
        """
        from src.core.holistic_estimator import HolisticEstimator
        
        with patch.object(HolisticEstimator, '__init__', lambda self, **kwargs: None):
            estimator = HolisticEstimator()
            estimator._holistic = Mock()
            estimator._holistic.process = Mock(return_value=MockHolisticResults(
                pose=True, left_hand=True, right_hand=True
            ))
            estimator._frame_count = 0
            estimator._detect_count = 0
            estimator._hand_detection_interval = 1
            estimator._last_hand_detection_frame = 0
            estimator._last_latency = 0
            estimator._adaptive_interval = False
            estimator._fallback_mode = False
            estimator._consecutive_hand_errors = 0
            estimator._max_hand_errors = 10
            estimator.enable_hands = True
            
            from src.core.holistic_estimator import HolisticEstimator as RealEstimator
            estimator._parse_pose_landmarks = RealEstimator._parse_pose_landmarks.__get__(estimator)
            estimator._parse_hand_landmarks = RealEstimator._parse_hand_landmarks.__get__(estimator)
            estimator._should_detect_hands = RealEstimator._should_detect_hands.__get__(estimator)
            estimator._adjust_detection_interval = RealEstimator._adjust_detection_interval.__get__(estimator)
            estimator.detect = RealEstimator.detect.__get__(estimator)
            estimator.detect_pose_only = RealEstimator.detect_pose_only.__get__(estimator)
            
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = estimator.detect_pose_only(image)
            
            assert isinstance(result, PoseResult)
            assert len(result.landmarks) == 33
    
    def test_pose_result_has_correct_structure(self):
        """
        **Property 2: API 向后兼容性**
        PoseResult 应有正确的结构（landmarks, detected, timestamp）
        """
        from src.core.holistic_estimator import HolisticEstimator
        
        with patch.object(HolisticEstimator, '__init__', lambda self, **kwargs: None):
            estimator = HolisticEstimator()
            estimator._holistic = Mock()
            estimator._holistic.process = Mock(return_value=MockHolisticResults(pose=True))
            estimator._frame_count = 0
            estimator._detect_count = 0
            estimator._hand_detection_interval = 1
            estimator._last_hand_detection_frame = 0
            estimator._last_latency = 0
            estimator._adaptive_interval = False
            estimator._fallback_mode = False
            estimator._consecutive_hand_errors = 0
            estimator._max_hand_errors = 10
            estimator.enable_hands = True
            
            from src.core.holistic_estimator import HolisticEstimator as RealEstimator
            estimator._parse_pose_landmarks = RealEstimator._parse_pose_landmarks.__get__(estimator)
            estimator._parse_hand_landmarks = RealEstimator._parse_hand_landmarks.__get__(estimator)
            estimator._should_detect_hands = RealEstimator._should_detect_hands.__get__(estimator)
            estimator._adjust_detection_interval = RealEstimator._adjust_detection_interval.__get__(estimator)
            estimator.detect = RealEstimator.detect.__get__(estimator)
            estimator.detect_pose_only = RealEstimator.detect_pose_only.__get__(estimator)
            
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = estimator.detect_pose_only(image, timestamp=123.456)
            
            assert hasattr(result, 'landmarks')
            assert hasattr(result, 'detected')
            assert hasattr(result, 'timestamp')
            assert result.detected == True
    
    def test_landmark_has_correct_attributes(self):
        """
        **Property 2: API 向后兼容性**
        每个 Landmark 应有 x, y, z, visibility 属性
        """
        from src.core.holistic_estimator import HolisticEstimator
        
        with patch.object(HolisticEstimator, '__init__', lambda self, **kwargs: None):
            estimator = HolisticEstimator()
            estimator._holistic = Mock()
            estimator._holistic.process = Mock(return_value=MockHolisticResults(pose=True))
            estimator._frame_count = 0
            estimator._detect_count = 0
            estimator._hand_detection_interval = 1
            estimator._last_hand_detection_frame = 0
            estimator._last_latency = 0
            estimator._adaptive_interval = False
            estimator._fallback_mode = False
            estimator._consecutive_hand_errors = 0
            estimator._max_hand_errors = 10
            estimator.enable_hands = True
            
            from src.core.holistic_estimator import HolisticEstimator as RealEstimator
            estimator._parse_pose_landmarks = RealEstimator._parse_pose_landmarks.__get__(estimator)
            estimator._parse_hand_landmarks = RealEstimator._parse_hand_landmarks.__get__(estimator)
            estimator._should_detect_hands = RealEstimator._should_detect_hands.__get__(estimator)
            estimator._adjust_detection_interval = RealEstimator._adjust_detection_interval.__get__(estimator)
            estimator.detect = RealEstimator.detect.__get__(estimator)
            estimator.detect_pose_only = RealEstimator.detect_pose_only.__get__(estimator)
            
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = estimator.detect_pose_only(image)
            
            for lm in result.landmarks:
                assert hasattr(lm, 'x')
                assert hasattr(lm, 'y')
                assert hasattr(lm, 'z')
                assert hasattr(lm, 'visibility')


class TestHandDetectionInterval:
    """手部检测间隔测试"""
    
    def test_set_hand_detection_interval(self):
        """设置手部检测间隔应生效"""
        from src.core.holistic_estimator import HolisticEstimator
        
        with patch.object(HolisticEstimator, '__init__', lambda self, **kwargs: None):
            estimator = HolisticEstimator()
            estimator._hand_detection_interval = 1
            
            from src.core.holistic_estimator import HolisticEstimator as RealEstimator
            estimator.set_hand_detection_interval = RealEstimator.set_hand_detection_interval.__get__(estimator)
            
            estimator.set_hand_detection_interval(5)
            assert estimator._hand_detection_interval == 5
    
    def test_interval_minimum_is_one(self):
        """检测间隔最小值应为1"""
        from src.core.holistic_estimator import HolisticEstimator
        
        with patch.object(HolisticEstimator, '__init__', lambda self, **kwargs: None):
            estimator = HolisticEstimator()
            estimator._hand_detection_interval = 1
            
            from src.core.holistic_estimator import HolisticEstimator as RealEstimator
            estimator.set_hand_detection_interval = RealEstimator.set_hand_detection_interval.__get__(estimator)
            
            estimator.set_hand_detection_interval(0)
            assert estimator._hand_detection_interval == 1
            
            estimator.set_hand_detection_interval(-5)
            assert estimator._hand_detection_interval == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
