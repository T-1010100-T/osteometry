"""
手部检测结果数据结构属性测试

**Feature: holistic-hand-integration, Property 4: 数据序列化往返一致性**
**Validates: Requirements 2.5, 10.1**
"""
import sys
from pathlib import Path
import math

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.pose_estimator import Landmark, PoseResult
from src.core.hand_result import HandResult, HolisticResult
from src.core.constants import (
    COMBINED_LANDMARK_INDICES,
    BODY_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    TOTAL_LANDMARK_COUNT,
)


# 生成有效的 Landmark 策略
landmark_strategy = st.builds(
    Landmark,
    x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    y=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    z=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    visibility=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
)

# 生成手部关键点列表策略 (21个点)
hand_landmarks_strategy = st.lists(
    landmark_strategy,
    min_size=21,
    max_size=21
)

# 生成身体关键点列表策略 (33个点)
body_landmarks_strategy = st.lists(
    landmark_strategy,
    min_size=33,
    max_size=33
)


class TestHandResultSerialization:
    """HandResult 序列化测试"""
    
    @given(
        landmarks=hand_landmarks_strategy,
        handedness=st.sampled_from(['left', 'right']),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_hand_result_round_trip(self, landmarks, handedness, confidence):
        """
        **Property 4: 数据序列化往返一致性**
        HandResult 序列化后反序列化应保持数据一致
        """
        original = HandResult(
            landmarks=landmarks,
            detected=True,
            handedness=handedness,
            confidence=confidence
        )
        
        # 序列化
        data = original.to_dict()
        
        # 反序列化
        restored = HandResult.from_dict(data)
        
        # 验证
        assert restored.detected == original.detected
        assert restored.handedness == original.handedness
        assert math.isclose(restored.confidence, original.confidence, rel_tol=1e-6)
        assert len(restored.landmarks) == len(original.landmarks)
        
        for orig_lm, rest_lm in zip(original.landmarks, restored.landmarks):
            assert math.isclose(rest_lm.x, orig_lm.x, rel_tol=1e-6)
            assert math.isclose(rest_lm.y, orig_lm.y, rel_tol=1e-6)
            assert math.isclose(rest_lm.z, orig_lm.z, rel_tol=1e-6)
            assert math.isclose(rest_lm.visibility, orig_lm.visibility, rel_tol=1e-6)



class TestHolisticResultSerialization:
    """HolisticResult 序列化测试"""
    
    @given(
        body_landmarks=body_landmarks_strategy,
        left_landmarks=hand_landmarks_strategy,
        right_landmarks=hand_landmarks_strategy,
        timestamp=st.floats(min_value=0.0, max_value=1e10, allow_nan=False)
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_holistic_result_json_round_trip(
        self, body_landmarks, left_landmarks, right_landmarks, timestamp
    ):
        """
        **Property 4: 数据序列化往返一致性**
        HolisticResult JSON序列化后反序列化应保持数据一致
        """
        original = HolisticResult(
            pose=PoseResult(landmarks=body_landmarks, detected=True, timestamp=timestamp),
            left_hand=HandResult(landmarks=left_landmarks, detected=True, handedness='left', confidence=0.9),
            right_hand=HandResult(landmarks=right_landmarks, detected=True, handedness='right', confidence=0.85),
            timestamp=timestamp
        )
        
        # JSON 序列化往返
        json_str = original.to_json()
        restored = HolisticResult.from_json(json_str)
        
        # 验证时间戳
        assert math.isclose(restored.timestamp, original.timestamp, rel_tol=1e-6)
        
        # 验证身体关键点
        assert len(restored.pose.landmarks) == len(original.pose.landmarks)
        for orig_lm, rest_lm in zip(original.pose.landmarks, restored.pose.landmarks):
            assert math.isclose(rest_lm.x, orig_lm.x, rel_tol=1e-6)
            assert math.isclose(rest_lm.y, orig_lm.y, rel_tol=1e-6)
        
        # 验证左手
        assert restored.left_hand.detected == original.left_hand.detected
        assert len(restored.left_hand.landmarks) == len(original.left_hand.landmarks)
        
        # 验证右手
        assert restored.right_hand.detected == original.right_hand.detected
        assert len(restored.right_hand.landmarks) == len(original.right_hand.landmarks)
    
    @given(
        body_landmarks=body_landmarks_strategy,
        left_detected=st.booleans(),
        right_detected=st.booleans()
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_holistic_result_dict_round_trip(
        self, body_landmarks, left_detected, right_detected
    ):
        """
        **Property 4: 数据序列化往返一致性**
        HolisticResult dict序列化后反序列化应保持数据一致
        """
        # 根据检测状态创建手部数据
        left_hand = HandResult(
            landmarks=[Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(21)] if left_detected else [],
            detected=left_detected,
            handedness='left',
            confidence=0.9 if left_detected else 0.0
        )
        right_hand = HandResult(
            landmarks=[Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(21)] if right_detected else [],
            detected=right_detected,
            handedness='right',
            confidence=0.85 if right_detected else 0.0
        )
        
        original = HolisticResult(
            pose=PoseResult(landmarks=body_landmarks, detected=True),
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp=123.456
        )
        
        # dict 序列化往返
        data = original.to_dict()
        restored = HolisticResult.from_dict(data)
        
        # 验证检测状态
        assert restored.left_hand.detected == original.left_hand.detected
        assert restored.right_hand.detected == original.right_hand.detected


class TestCombinedLandmarks:
    """合并关键点测试"""
    
    def test_combined_landmarks_length(self):
        """合并关键点列表应有75个元素"""
        body_landmarks = [Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(33)]
        left_landmarks = [Landmark(0.3, 0.3, 0.0, 0.9) for _ in range(21)]
        right_landmarks = [Landmark(0.7, 0.7, 0.0, 0.9) for _ in range(21)]
        
        result = HolisticResult(
            pose=PoseResult(landmarks=body_landmarks, detected=True),
            left_hand=HandResult(landmarks=left_landmarks, detected=True, handedness='left'),
            right_hand=HandResult(landmarks=right_landmarks, detected=True, handedness='right')
        )
        
        combined = result.get_combined_landmarks()
        assert len(combined) == TOTAL_LANDMARK_COUNT
    
    def test_combined_landmarks_body_indices(self):
        """身体关键点应在索引 0-32"""
        body_landmarks = [Landmark(i / 33.0, 0.5, 0.0, 0.9) for i in range(33)]
        
        result = HolisticResult(
            pose=PoseResult(landmarks=body_landmarks, detected=True),
            left_hand=HandResult.empty('left'),
            right_hand=HandResult.empty('right')
        )
        
        combined = result.get_combined_landmarks()
        
        # 验证身体关键点位置
        for i in range(33):
            assert combined[i] is not None
            assert math.isclose(combined[i].x, i / 33.0, rel_tol=1e-6)
    
    def test_combined_landmarks_hand_indices(self):
        """
        **Property 3: 合并索引映射正确性**
        左手应在33-53，右手应在54-74
        """
        left_landmarks = [Landmark(0.1, 0.0, 0.0, 0.9) for _ in range(21)]
        right_landmarks = [Landmark(0.9, 0.0, 0.0, 0.9) for _ in range(21)]
        
        result = HolisticResult(
            pose=PoseResult(landmarks=[Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(33)], detected=True),
            left_hand=HandResult(landmarks=left_landmarks, detected=True, handedness='left'),
            right_hand=HandResult(landmarks=right_landmarks, detected=True, handedness='right')
        )
        
        combined = result.get_combined_landmarks()
        
        # 验证左手位置 (33-53)
        for i in range(33, 54):
            assert combined[i] is not None
            assert math.isclose(combined[i].x, 0.1, rel_tol=1e-6)
        
        # 验证右手位置 (54-74)
        for i in range(54, 75):
            assert combined[i] is not None
            assert math.isclose(combined[i].x, 0.9, rel_tol=1e-6)
    
    def test_combined_landmarks_undetected_hands(self):
        """未检测到的手部位置应为 None"""
        body_landmarks = [Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(33)]
        
        result = HolisticResult(
            pose=PoseResult(landmarks=body_landmarks, detected=True),
            left_hand=HandResult.empty('left'),
            right_hand=HandResult.empty('right')
        )
        
        combined = result.get_combined_landmarks()
        
        # 身体应有值
        for i in range(33):
            assert combined[i] is not None
        
        # 手部应为 None
        for i in range(33, 75):
            assert combined[i] is None


class TestBackwardCompatibility:
    """向后兼容性测试"""
    
    def test_to_pose_result_returns_body_only(self):
        """to_pose_result 应仅返回身体关键点"""
        body_landmarks = [Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(33)]
        left_landmarks = [Landmark(0.3, 0.3, 0.0, 0.9) for _ in range(21)]
        
        holistic = HolisticResult(
            pose=PoseResult(landmarks=body_landmarks, detected=True, timestamp=100.0),
            left_hand=HandResult(landmarks=left_landmarks, detected=True, handedness='left')
        )
        
        pose_result = holistic.to_pose_result()
        
        assert isinstance(pose_result, PoseResult)
        assert len(pose_result.landmarks) == 33
        assert pose_result.detected == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
