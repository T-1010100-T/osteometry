"""
智能数据采集器类型测试

**Feature: smart-data-collector**
**Property: Data model round-trip**
**Validates: Requirements 4.2**
"""
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from src.core.smart_collector_types import (
    CollectorState,
    StabilityResult,
    QualityResult,
    FusionResult,
    CollectorStatus,
    SessionData,
    CollectorConfig
)


# ============== Strategies ==============

@st.composite
def stability_result_strategy(draw):
    """生成随机 StabilityResult"""
    return StabilityResult(
        is_stable=draw(st.booleans()),
        progress=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        body_stable=draw(st.booleans()),
        left_hand_stable=draw(st.booleans()),
        right_hand_stable=draw(st.booleans()),
        body_movement=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        hand_movement=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        stable_frames=draw(st.integers(min_value=0, max_value=100))
    )


@st.composite
def quality_result_strategy(draw):
    """生成随机 QualityResult"""
    return QualityResult(
        score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        confidence_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        stability_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        completeness_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        recommendation=draw(st.sampled_from(['auto_save', 'confirm', 'retry'])),
        issues=draw(st.lists(st.text(min_size=0, max_size=50), max_size=5)),
        anatomy_valid=draw(st.booleans()),
        anatomy_issues=draw(st.lists(st.text(min_size=0, max_size=50), max_size=5))
    )


@st.composite
def fusion_result_strategy(draw):
    """生成随机 FusionResult"""
    return FusionResult(
        body_measurement=draw(st.none() | st.fixed_dictionaries({
            'height': st.floats(min_value=0.5, max_value=2.5, allow_nan=False)
        })),
        left_hand=draw(st.none() | st.fixed_dictionaries({
            'palm_length': st.floats(min_value=0.05, max_value=0.15, allow_nan=False)
        })),
        right_hand=draw(st.none() | st.fixed_dictionaries({
            'palm_length': st.floats(min_value=0.05, max_value=0.15, allow_nan=False)
        })),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        frames_used=draw(st.integers(min_value=0, max_value=10)),
        outliers_removed=draw(st.integers(min_value=0, max_value=10)),
        fusion_timestamp=draw(st.floats(min_value=0.0, max_value=1e10, allow_nan=False))
    )


@st.composite
def collector_status_strategy(draw):
    """生成随机 CollectorStatus"""
    return CollectorStatus(
        state=draw(st.sampled_from(list(CollectorState))),
        stability_progress=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        quality_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        message=draw(st.text(min_size=0, max_size=100)),
        saved_path=draw(st.none() | st.text(min_size=1, max_size=100)),
        region_status=draw(st.fixed_dictionaries({
            'body': st.booleans(),
            'left_hand': st.booleans(),
            'right_hand': st.booleans()
        }))
    )


@st.composite
def collector_config_strategy(draw):
    """生成随机 CollectorConfig"""
    return CollectorConfig(
        stability_window=draw(st.integers(min_value=1, max_value=30)),
        body_threshold=draw(st.floats(min_value=0.001, max_value=0.1, allow_nan=False)),
        hand_threshold=draw(st.floats(min_value=0.001, max_value=0.1, allow_nan=False)),
        fusion_frames=draw(st.integers(min_value=1, max_value=20)),
        auto_save_threshold=draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False)),
        confirm_threshold=draw(st.floats(min_value=0.3, max_value=0.8, allow_nan=False)),
        confidence_weight=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        stability_weight=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        completeness_weight=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        output_dir=draw(st.text(min_size=1, max_size=50))
    )


# ============== Property Tests ==============

class TestDataModelRoundTrip:
    """
    Property: Data model round-trip
    For any data model instance, serializing to dict and back should produce equivalent object
    """
    
    @given(stability_result_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_stability_result_round_trip(self, original: StabilityResult):
        """StabilityResult 序列化往返测试"""
        # Serialize to dict
        data = original.to_dict()
        # Deserialize back
        restored = StabilityResult.from_dict(data)
        # Verify equality
        assert restored.is_stable == original.is_stable
        assert restored.progress == original.progress
        assert restored.body_stable == original.body_stable
        assert restored.stable_frames == original.stable_frames
    
    @given(quality_result_strategy())
    @settings(max_examples=100)
    def test_quality_result_round_trip(self, original: QualityResult):
        """QualityResult 序列化往返测试"""
        data = original.to_dict()
        restored = QualityResult.from_dict(data)
        assert restored.score == original.score
        assert restored.recommendation == original.recommendation
        assert restored.anatomy_valid == original.anatomy_valid
    
    @given(fusion_result_strategy())
    @settings(max_examples=100)
    def test_fusion_result_round_trip(self, original: FusionResult):
        """FusionResult 序列化往返测试"""
        data = original.to_dict()
        restored = FusionResult.from_dict(data)
        assert restored.confidence == original.confidence
        assert restored.frames_used == original.frames_used
        assert restored.body_measurement == original.body_measurement
    
    @given(collector_status_strategy())
    @settings(max_examples=100)
    def test_collector_status_round_trip(self, original: CollectorStatus):
        """CollectorStatus 序列化往返测试"""
        data = original.to_dict()
        restored = CollectorStatus.from_dict(data)
        assert restored.state == original.state
        assert restored.stability_progress == original.stability_progress
        assert restored.quality_score == original.quality_score
    
    @given(collector_config_strategy())
    @settings(max_examples=100)
    def test_collector_config_round_trip(self, original: CollectorConfig):
        """CollectorConfig 序列化往返测试"""
        data = original.to_dict()
        restored = CollectorConfig.from_dict(data)
        assert restored.stability_window == original.stability_window
        assert restored.body_threshold == original.body_threshold
        assert restored.fusion_frames == original.fusion_frames


class TestCollectorState:
    """CollectorState 枚举测试"""
    
    def test_all_states_have_string_value(self):
        """所有状态都有字符串值"""
        for state in CollectorState:
            assert isinstance(state.value, str)
            assert len(state.value) > 0
    
    def test_state_from_string(self):
        """可以从字符串创建状态"""
        assert CollectorState("idle") == CollectorState.IDLE
        assert CollectorState("monitoring") == CollectorState.MONITORING
        assert CollectorState("completed") == CollectorState.COMPLETED


class TestSessionData:
    """SessionData 测试"""
    
    def test_json_round_trip(self):
        """JSON 序列化往返测试"""
        original = SessionData(
            session_id="test_123",
            start_time="2024-01-15T10:00:00",
            measurements=[{"height": 1.75}],
            metadata={"user": "test"}
        )
        json_str = original.to_json()
        restored = SessionData.from_json(json_str)
        assert restored.session_id == original.session_id
        assert restored.measurements == original.measurements
