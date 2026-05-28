"""
数据导出属性测试

**Feature: holistic-hand-integration, Property 16: CSV导出格式正确性**
**Validates: Requirements 10.5**
"""
import sys
from pathlib import Path
import tempfile
import csv
import json

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.output.data_exporter import HolisticJSONExporter, HolisticCSVExporter


def generate_test_frame(frame_id: int, num_body_points: int = 33, 
                        left_detected: bool = True, right_detected: bool = True) -> dict:
    """生成测试帧数据"""
    frame = {
        'frame_id': frame_id,
        'timestamp': frame_id * 0.033,  # ~30fps
        'body_points': [
            {'x': 0.5, 'y': 0.5, 'z': 1.0, 'confidence': 0.9}
            for _ in range(num_body_points)
        ],
        'hands': {
            'left': {
                'detected': left_detected,
                'points_2d': [
                    {'x': 0.3, 'y': 0.6, 'z': 1.0, 'confidence': 0.85}
                    for _ in range(21)
                ] if left_detected else [],
                'confidence': 0.85 if left_detected else 0.0
            },
            'right': {
                'detected': right_detected,
                'points_2d': [
                    {'x': 0.7, 'y': 0.6, 'z': 1.0, 'confidence': 0.85}
                    for _ in range(21)
                ] if right_detected else [],
                'confidence': 0.85 if right_detected else 0.0
            }
        }
    }
    return frame


class TestCSVExportFormat:
    """
    CSV导出格式正确性测试
    
    **Feature: holistic-hand-integration, Property 16: CSV导出格式正确性**
    **Validates: Requirements 10.5**
    """
    
    @given(
        num_frames=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_csv_has_correct_row_count(self, num_frames: int):
        """
        **Feature: holistic-hand-integration, Property 16: CSV导出格式正确性**
        **Validates: Requirements 10.5**
        
        *For any* session with N frames, the exported CSV SHALL have N rows (plus header)
        """
        exporter = HolisticCSVExporter()
        
        # 生成测试帧
        frames = [generate_test_frame(i) for i in range(num_frames)]
        
        # 导出到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            result = exporter.export_holistic_batch(frames, filepath)
            assert result == True
            
            # 读取CSV并验证行数
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # 应该有 header + N 行数据
            assert len(rows) == num_frames + 1
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_csv_has_correct_column_count(self):
        """
        **Feature: holistic-hand-integration, Property 16: CSV导出格式正确性**
        **Validates: Requirements 10.5**
        
        Each row SHALL have columns for all 75 landmark coordinates (225 coordinate values + metadata)
        """
        exporter = HolisticCSVExporter()
        
        frames = [generate_test_frame(0)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            exporter.export_holistic_batch(frames, filepath)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                data_row = next(reader)
            
            # 计算预期列数:
            # 2 (frame_id, timestamp) + 
            # 33*3 (body) + 21*3 (left hand) + 21*3 (right hand) + 
            # 4 (metadata: body_detected, left_hand_detected, right_hand_detected, partial_hand_data)
            # = 2 + 99 + 63 + 63 + 4 = 231
            expected_columns = 2 + 33*3 + 21*3 + 21*3 + 4
            
            assert len(header) == expected_columns
            assert len(data_row) == expected_columns
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_csv_header_format(self):
        """
        **Feature: holistic-hand-integration, Property 16: CSV导出格式正确性**
        **Validates: Requirements 10.5**
        
        CSV header SHALL contain proper column names for all landmarks
        """
        exporter = HolisticCSVExporter()
        
        frames = [generate_test_frame(0)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            exporter.export_holistic_batch(frames, filepath)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
            
            # 验证基本列
            assert 'frame_id' in header
            assert 'timestamp' in header
            
            # 验证身体关键点列
            assert 'body_0_x' in header
            assert 'body_0_y' in header
            assert 'body_0_z' in header
            assert 'body_32_x' in header
            
            # 验证左手关键点列
            assert 'left_hand_0_x' in header
            assert 'left_hand_20_x' in header
            
            # 验证右手关键点列
            assert 'right_hand_0_x' in header
            assert 'right_hand_20_x' in header
            
            # 验证元数据列
            assert 'body_detected' in header
            assert 'left_hand_detected' in header
            assert 'right_hand_detected' in header
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    @given(
        left_detected=st.booleans(),
        right_detected=st.booleans()
    )
    @settings(max_examples=10)
    def test_csv_handles_partial_hand_data(self, left_detected: bool, right_detected: bool):
        """
        **Feature: holistic-hand-integration, Property 16: CSV导出格式正确性**
        **Validates: Requirements 10.5**
        
        CSV SHALL correctly record hand detection status
        """
        exporter = HolisticCSVExporter()
        
        frames = [generate_test_frame(0, left_detected=left_detected, right_detected=right_detected)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            exporter.export_holistic_batch(frames, filepath)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                data_row = next(reader)
            
            # 找到检测状态列的索引
            left_idx = header.index('left_hand_detected')
            right_idx = header.index('right_hand_detected')
            
            # 验证检测状态
            assert data_row[left_idx] == ('1' if left_detected else '0')
            assert data_row[right_idx] == ('1' if right_detected else '0')
            
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestJSONExportFormat:
    """JSON导出格式测试"""
    
    def test_json_frame_structure(self):
        """JSON帧数据应包含正确的结构"""
        from src.core.hand_result import HandResult, HolisticResult
        from src.core.pose_estimator import PoseResult, Landmark
        
        exporter = HolisticJSONExporter()
        
        # 创建测试数据
        landmarks = [Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(33)]
        pose_result = PoseResult(landmarks=landmarks, detected=True)
        
        hand_landmarks = [Landmark(0.5, 0.5, 0.0, 0.9) for _ in range(21)]
        left_hand = HandResult(landmarks=hand_landmarks, detected=True, 
                              handedness='left', confidence=0.9)
        right_hand = HandResult.empty('right')
        
        holistic_result = HolisticResult(
            pose=pose_result,
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp=0.0
        )
        
        # 导出帧数据
        frame_data = exporter.export_frame(
            frame_id=1,
            timestamp=0.033,
            holistic_result=holistic_result
        )
        
        # 验证结构
        assert 'frame_id' in frame_data
        assert 'timestamp' in frame_data
        assert 'body_points' in frame_data
        assert 'hands' in frame_data
        assert 'left' in frame_data['hands']
        assert 'right' in frame_data['hands']
        
        # 验证身体点数量
        assert len(frame_data['body_points']) == 33
        
        # 验证左手数据
        assert frame_data['hands']['left']['detected'] == True
        
        # 验证右手数据
        assert frame_data['hands']['right']['detected'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
