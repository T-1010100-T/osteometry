"""
会话管理器测试

**Feature: smart-data-collector**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings, HealthCheck

from src.core.session_manager import SessionManager
from src.core.smart_collector_types import FusionResult, QualityResult


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


# ============== Property Tests ==============

class TestSessionIDUniqueness:
    """
    Property 8: Session ID Uniqueness
    For any two sessions, their IDs should be different
    """
    
    def test_multiple_sessions_unique_ids(self, temp_dir):
        """多个会话ID唯一"""
        manager = SessionManager(output_dir=temp_dir, auto_save=False)
        
        session_ids = set()
        for _ in range(10):
            session_id = manager.start_session()
            assert session_id not in session_ids, "会话ID重复"
            session_ids.add(session_id)
            manager.end_session()
        
        assert len(session_ids) == 10
    
    @given(st.integers(min_value=5, max_value=20))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_session_ids_always_unique(self, num_sessions: int):
        """会话ID始终唯一"""
        import tempfile
        temp_path = tempfile.mkdtemp()
        try:
            manager = SessionManager(output_dir=temp_path, auto_save=False)
            
            session_ids = []
            for _ in range(num_sessions):
                session_id = manager.start_session()
                session_ids.append(session_id)
                manager.end_session()
            
            # 所有ID应该唯一
            assert len(session_ids) == len(set(session_ids))
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


class TestSessionLifecycle:
    """会话生命周期测试"""
    
    def test_start_session_creates_id(self, temp_dir):
        """开始会话创建ID"""
        manager = SessionManager(output_dir=temp_dir, auto_save=False)
        
        session_id = manager.start_session()
        
        assert session_id is not None
        assert len(session_id) > 0
        assert manager.has_active_session is True
    
    def test_end_session_returns_summary(self, temp_dir):
        """结束会话返回摘要"""
        manager = SessionManager(output_dir=temp_dir, auto_save=False)
        
        manager.start_session()
        
        # 添加一些测量数据
        fusion = FusionResult(
            body_measurement={'height': 1.75},
            confidence=0.85,
            frames_used=5
        )
        quality = QualityResult(score=0.85, recommendation='auto_save')
        manager.add_measurement(fusion, quality)
        
        summary = manager.end_session()
        
        assert summary is not None
        assert 'total_measurements' in summary
        assert summary['total_measurements'] == 1
    
    def test_session_without_end(self, temp_dir):
        """未结束的会话"""
        manager = SessionManager(output_dir=temp_dir, auto_save=False)
        
        manager.start_session()
        assert manager.has_active_session is True
        
        # 开始新会话会自动结束旧会话吗？不会，需要显式结束
        # 这里测试可以添加测量
        fusion = FusionResult(body_measurement={'height': 1.75})
        manager.add_measurement(fusion)
        
        assert manager.has_active_session is True


class TestMeasurementStorage:
    """测量数据存储测试"""
    
    def test_add_measurement_to_session(self, temp_dir):
        """添加测量到会话"""
        manager = SessionManager(output_dir=temp_dir, auto_save=True)
        
        manager.start_session()
        
        fusion = FusionResult(
            body_measurement={'height': 1.75, 'shoulder_width': 0.45},
            left_hand={'palm_length': 0.08},
            confidence=0.9,
            frames_used=5
        )
        quality = QualityResult(score=0.88, recommendation='auto_save')
        
        file_path = manager.add_measurement(fusion, quality)
        
        assert file_path is not None
        assert Path(file_path).exists()
    
    def test_multiple_measurements_in_session(self, temp_dir):
        """会话中多次测量"""
        manager = SessionManager(output_dir=temp_dir, auto_save=False)
        
        manager.start_session()
        
        for i in range(3):
            fusion = FusionResult(
                body_measurement={'height': 1.75 + i * 0.01},
                confidence=0.8 + i * 0.05
            )
            manager.add_measurement(fusion)
        
        summary = manager.end_session()
        
        assert summary['total_measurements'] == 3
    
    def test_auto_start_session_on_measurement(self, temp_dir):
        """添加测量时自动开始会话"""
        manager = SessionManager(output_dir=temp_dir, auto_save=False)
        
        assert manager.has_active_session is False
        
        fusion = FusionResult(body_measurement={'height': 1.75})
        manager.add_measurement(fusion)
        
        assert manager.has_active_session is True


class TestSessionSummary:
    """会话摘要测试"""
    
    def test_summary_statistics(self, temp_dir):
        """摘要统计计算"""
        manager = SessionManager(output_dir=temp_dir, auto_save=False)
        
        manager.start_session()
        
        # 添加多个测量
        for score in [0.7, 0.8, 0.9]:
            fusion = FusionResult(
                body_measurement={'height': 1.75},
                confidence=score
            )
            quality = QualityResult(score=score)
            manager.add_measurement(fusion, quality)
        
        summary = manager.end_session()
        
        assert 'quality_stats' in summary
        assert summary['quality_stats']['mean'] == pytest.approx(0.8, rel=0.01)
        assert summary['quality_stats']['min'] == pytest.approx(0.7, rel=0.01)
        assert summary['quality_stats']['max'] == pytest.approx(0.9, rel=0.01)


class TestSessionHistory:
    """会话历史测试"""
    
    def test_get_session_history(self, temp_dir):
        """获取会话历史"""
        manager = SessionManager(output_dir=temp_dir, auto_save=True)
        
        # 创建几个会话
        for i in range(3):
            manager.start_session(metadata={'index': i})
            fusion = FusionResult(body_measurement={'height': 1.75})
            manager.add_measurement(fusion)
            manager.end_session()
        
        history = manager.get_session_history(limit=10)
        
        assert len(history) == 3
    
    def test_load_session(self, temp_dir):
        """加载会话"""
        manager = SessionManager(output_dir=temp_dir, auto_save=True)
        
        session_id = manager.start_session(metadata={'test': True})
        fusion = FusionResult(body_measurement={'height': 1.75})
        manager.add_measurement(fusion)
        manager.end_session()
        
        # 加载会话
        loaded = manager.load_session(session_id)
        
        assert loaded is not None
        assert loaded.session_id == session_id
        assert len(loaded.measurements) == 1


class TestFileNaming:
    """文件命名测试"""
    
    def test_session_file_naming(self, temp_dir):
        """会话文件命名格式"""
        manager = SessionManager(output_dir=temp_dir, auto_save=True)
        
        session_id = manager.start_session()
        manager.end_session()
        
        # 检查文件存在
        session_file = Path(temp_dir) / f"{session_id}.json"
        assert session_file.exists()
    
    def test_measurement_file_naming(self, temp_dir):
        """测量文件命名格式：YYYY-MM-DD HH-MM-SS"""
        manager = SessionManager(output_dir=temp_dir, auto_save=True)
        
        manager.start_session()
        fusion = FusionResult(body_measurement={'height': 1.75})
        quality = QualityResult(score=0.85)
        
        file_path = manager.add_measurement(fusion, quality, measurement_type='full_body')
        
        # 检查文件名格式为时间格式 YYYY-MM-DD HH-MM-SS
        import re
        filename = Path(file_path).stem
        # 匹配格式: 2025-12-15 17-29-35
        assert re.match(r'\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}', filename)
