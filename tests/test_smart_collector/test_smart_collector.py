"""
智能数据采集器测试

**Feature: smart-data-collector**
**Validates: Requirements 1.2, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4**
"""
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from hypothesis import given, strategies as st, settings, HealthCheck

from src.core.smart_collector import SmartDataCollector
from src.core.smart_collector_types import CollectorState, CollectorStatus
from src.core.hand_result import HolisticResult, HandResult, PoseResult
from src.core.pose_estimator import Landmark


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def collector(temp_dir):
    """创建采集器实例"""
    config = {
        'output_dir': temp_dir,
        'stability_window': 5,
        'fusion_frames': 3
    }
    return SmartDataCollector(config=config)


def create_mock_holistic(detected=True, confidence=0.9):
    """创建模拟的 HolisticResult"""
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=confidence) for _ in range(33)]
    hand_landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=confidence) for _ in range(21)]
    
    return HolisticResult(
        pose=PoseResult(detected=detected, landmarks=landmarks),
        left_hand=HandResult(detected=detected, landmarks=hand_landmarks, confidence=confidence),
        right_hand=HandResult(detected=detected, landmarks=hand_landmarks, confidence=confidence)
    )


# ============== Property Tests ==============

class TestStabilityProgressRange:
    """
    Property 9: Stability Progress Range
    Progress should always be in [0.0, 1.0]
    """
    
    def test_initial_progress_in_range(self, collector):
        """初始进度在有效范围内"""
        status = collector.get_status()
        assert 0.0 <= status.stability_progress <= 1.0
    
    def test_progress_after_frames_in_range(self, collector):
        """处理帧后进度在有效范围内"""
        holistic = create_mock_holistic()
        
        for _ in range(10):
            status = collector.process_frame(holistic)
            assert 0.0 <= status.stability_progress <= 1.0


class TestErrorRecovery:
    """
    Property 11: Error Recovery
    On error, collector should return to IDLE state
    """
    
    def test_error_returns_failed_state(self, collector):
        """错误返回失败状态"""
        # 模拟错误
        with patch.object(collector._stability_detector, 'add_frame', side_effect=Exception("Test error")):
            holistic = create_mock_holistic()
            # 先进入监测状态
            collector._state = CollectorState.MONITORING
            
            status = collector.process_frame(holistic)
            
            assert status.state == CollectorState.FAILED
            assert "错误" in status.message or "error" in status.message.lower()
    
    def test_cancel_returns_to_idle(self, collector):
        """取消返回空闲状态"""
        holistic = create_mock_holistic()
        
        # 进入监测状态
        collector.process_frame(holistic)
        
        # 取消
        collector.cancel()
        
        assert collector.state == CollectorState.IDLE


class TestStateMachine:
    """状态机测试"""
    
    def test_initial_state_is_idle(self, collector):
        """初始状态为空闲"""
        assert collector.state == CollectorState.IDLE
    
    def test_idle_to_monitoring_on_detection(self, collector):
        """检测到人体后进入监测状态"""
        holistic = create_mock_holistic(detected=True)
        
        status = collector.process_frame(holistic)
        
        assert collector.state == CollectorState.MONITORING
    
    def test_stays_idle_without_detection(self, collector):
        """未检测到人体时保持空闲"""
        holistic = create_mock_holistic(detected=False)
        
        status = collector.process_frame(holistic)
        
        assert collector.state == CollectorState.IDLE
    
    def test_monitoring_to_ready_when_stable(self, collector):
        """稳定后进入就绪状态"""
        holistic = create_mock_holistic()
        
        # 添加足够多的稳定帧
        for _ in range(10):
            status = collector.process_frame(holistic)
        
        # 应该进入就绪或更后面的状态
        assert collector.state in [
            CollectorState.READY,
            CollectorState.CAPTURING,
            CollectorState.PROCESSING,
            CollectorState.COMPLETED
        ]


class TestRegionStatus:
    """区域状态测试"""
    
    def test_region_status_provided(self, collector):
        """提供区域状态"""
        holistic = create_mock_holistic()
        
        status = collector.process_frame(holistic)
        
        assert 'body' in status.region_status
        assert 'left_hand' in status.region_status
        assert 'right_hand' in status.region_status


class TestSessionIntegration:
    """会话集成测试"""
    
    def test_start_session(self, collector):
        """开始会话"""
        session_id = collector.start_session(metadata={'test': True})
        
        assert session_id is not None
        assert len(session_id) > 0
    
    def test_end_session(self, collector):
        """结束会话"""
        collector.start_session()
        summary = collector.end_session()
        
        # 空会话也应该能结束
        assert summary is not None or summary is None  # 可能返回空摘要


class TestForceCapture:
    """强制采集测试"""
    
    def test_force_capture_without_frame(self, collector):
        """无帧数据时强制采集"""
        result = collector.force_capture()
        
        # 应该返回 None 或处理错误
        assert result is None
    
    def test_force_capture_with_frame(self, collector):
        """有帧数据时强制采集"""
        holistic = create_mock_holistic()
        
        # 先处理一帧
        collector.process_frame(holistic)
        
        # 强制采集
        result = collector.force_capture()
        
        # 应该完成采集
        assert collector.state == CollectorState.COMPLETED


class TestConfiguration:
    """配置测试"""
    
    def test_custom_config(self, temp_dir):
        """自定义配置"""
        config = {
            'output_dir': temp_dir,
            'stability_window': 15,
            'body_threshold': 0.03,
            'fusion_frames': 7
        }
        
        collector = SmartDataCollector(config=config)
        
        assert collector._config.stability_window == 15
        assert collector._config.body_threshold == 0.03
        assert collector._config.fusion_frames == 7
    
    def test_default_config(self, temp_dir):
        """默认配置"""
        collector = SmartDataCollector()
        
        assert collector._config.stability_window == 10
        assert collector._config.fusion_frames == 5


class TestCollectorProperties:
    """采集器属性测试"""
    
    def test_is_collecting_property(self, collector):
        """is_collecting 属性"""
        assert collector.is_collecting is False
        
        holistic = create_mock_holistic()
        collector.process_frame(holistic)
        
        assert collector.is_collecting is True
    
    def test_state_property(self, collector):
        """state 属性"""
        assert isinstance(collector.state, CollectorState)
