"""
手部常量定义属性测试

**Feature: holistic-hand-integration, Property 3: 合并索引映射正确性**
**Validates: Requirements 2.3**
"""
import sys
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.constants import (
    HandLandmark,
    HAND_LANDMARK_NAMES,
    HAND_LANDMARK_INDICES,
    HAND_CONNECTIONS,
    COMBINED_LANDMARK_INDICES,
    BODY_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT,
    TOTAL_LANDMARK_COUNT,
    FINGER_LANDMARKS,
    FINGERTIP_LANDMARKS,
    MCP_LANDMARKS,
)


class TestHandLandmarkConstants:
    """手部关键点常量测试"""
    
    def test_hand_landmark_count(self):
        """手部关键点应有21个"""
        assert len(HandLandmark) == 21
        assert HAND_LANDMARK_COUNT == 21
    
    def test_hand_landmark_names_complete(self):
        """所有手部关键点都应有名称映射"""
        assert len(HAND_LANDMARK_NAMES) == 21
        for i in range(21):
            assert i in HAND_LANDMARK_NAMES
    
    def test_hand_landmark_indices_reverse_mapping(self):
        """名称到索引的反向映射应正确"""
        for idx, name in HAND_LANDMARK_NAMES.items():
            assert HAND_LANDMARK_INDICES[name] == idx
    
    def test_finger_landmarks_complete(self):
        """每个手指应有4个关键点"""
        assert len(FINGER_LANDMARKS) == 5
        for finger, landmarks in FINGER_LANDMARKS.items():
            assert len(landmarks) == 4, f"{finger} should have 4 landmarks"
    
    def test_fingertip_landmarks_count(self):
        """应有5个指尖关键点"""
        assert len(FINGERTIP_LANDMARKS) == 5
    
    def test_mcp_landmarks_count(self):
        """应有4个MCP关键点（不含拇指）"""
        assert len(MCP_LANDMARKS) == 4


class TestCombinedLandmarkIndices:
    """
    合并索引映射正确性测试
    
    **Property 3: 合并索引映射正确性**
    """
    
    def test_body_index_range(self):
        """
        **Property 3: 合并索引映射正确性**
        身体关键点应在索引 0-32
        """
        start, end = COMBINED_LANDMARK_INDICES['body']
        assert start == 0
        assert end == 33
        assert end - start == BODY_LANDMARK_COUNT
    
    def test_left_hand_index_range(self):
        """
        **Property 3: 合并索引映射正确性**
        左手关键点应在索引 33-53
        """
        start, end = COMBINED_LANDMARK_INDICES['left_hand']
        assert start == 33
        assert end == 54
        assert end - start == HAND_LANDMARK_COUNT
    
    def test_right_hand_index_range(self):
        """
        **Property 3: 合并索引映射正确性**
        右手关键点应在索引 54-74
        """
        start, end = COMBINED_LANDMARK_INDICES['right_hand']
        assert start == 54
        assert end == 75
        assert end - start == HAND_LANDMARK_COUNT
    
    def test_total_landmark_count(self):
        """
        **Property 3: 合并索引映射正确性**
        总关键点数应为75 (33 + 21 + 21)
        """
        assert TOTAL_LANDMARK_COUNT == 75
        assert TOTAL_LANDMARK_COUNT == BODY_LANDMARK_COUNT + 2 * HAND_LANDMARK_COUNT
    
    def test_index_ranges_non_overlapping(self):
        """
        **Property 3: 合并索引映射正确性**
        各部分索引范围不应重叠
        """
        body_start, body_end = COMBINED_LANDMARK_INDICES['body']
        left_start, left_end = COMBINED_LANDMARK_INDICES['left_hand']
        right_start, right_end = COMBINED_LANDMARK_INDICES['right_hand']
        
        # 身体结束 == 左手开始
        assert body_end == left_start
        # 左手结束 == 右手开始
        assert left_end == right_start
        # 右手结束 == 总数
        assert right_end == TOTAL_LANDMARK_COUNT
    
    def test_index_ranges_contiguous(self):
        """
        **Property 3: 合并索引映射正确性**
        索引范围应连续无间隙
        """
        all_indices = set()
        for part, (start, end) in COMBINED_LANDMARK_INDICES.items():
            for i in range(start, end):
                assert i not in all_indices, f"Index {i} is duplicated"
                all_indices.add(i)
        
        # 应覆盖 0 到 74 的所有索引
        assert all_indices == set(range(TOTAL_LANDMARK_COUNT))
    
    @given(st.integers(min_value=0, max_value=74))
    @settings(max_examples=75)
    def test_any_index_belongs_to_exactly_one_part(self, index: int):
        """
        **Property 3: 合并索引映射正确性**
        对于任意有效索引，应恰好属于一个部分（身体/左手/右手）
        """
        belonging_parts = []
        for part, (start, end) in COMBINED_LANDMARK_INDICES.items():
            if start <= index < end:
                belonging_parts.append(part)
        
        assert len(belonging_parts) == 1, f"Index {index} belongs to {len(belonging_parts)} parts"


class TestHandConnections:
    """手部骨骼连接测试"""
    
    def test_connections_valid_indices(self):
        """所有连接的索引应在有效范围内"""
        for start, end in HAND_CONNECTIONS:
            assert 0 <= start < 21
            assert 0 <= end < 21
    
    def test_connections_count(self):
        """手部骨骼连接数应正确"""
        # 拇指3段 + 其他4指各3段 + 手腕到5个指根 + 3个MCP连接 = 3 + 12 + 5 + 3 = 23
        assert len(HAND_CONNECTIONS) == 23


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
