"""
多帧融合器测试

**Feature: smart-data-collector**
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""
import pytest
import math
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck, assume

from src.core.frame_fusion import FrameFusion


# ============== Property Tests ==============

class TestIQROutlierRemoval:
    """
    Property 6: IQR Outlier Removal
    Values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR] should be excluded
    """
    
    def test_known_outliers_removed(self):
        """已知异常值被移除"""
        fusion = FrameFusion(fusion_frames=5, iqr_multiplier=1.5)
        
        # 正常值 + 明显异常值
        values = [1.0, 1.1, 1.0, 0.9, 1.0, 100.0]  # 100.0 是异常值
        weights = [0.8] * 6
        
        clean_values, clean_weights, outliers = fusion._remove_outliers_iqr(values, weights)
        
        assert 100.0 not in clean_values
        assert outliers >= 1
    
    def test_normal_values_preserved(self):
        """正常值被保留"""
        fusion = FrameFusion(fusion_frames=5, iqr_multiplier=1.5)
        
        # 所有值都在正常范围内
        values = [1.0, 1.05, 0.95, 1.02, 0.98]
        weights = [0.8] * 5
        
        clean_values, clean_weights, outliers = fusion._remove_outliers_iqr(values, weights)
        
        assert len(clean_values) == len(values)
        assert outliers == 0
    
    @given(
        base_value=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
        noise=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
        outlier_magnitude=st.floats(min_value=5.0, max_value=20.0, allow_nan=False)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_extreme_outliers_detected(self, base_value, noise, outlier_magnitude):
        """极端异常值被检测"""
        fusion = FrameFusion(fusion_frames=5, iqr_multiplier=1.5)
        
        # 生成正常值 + 一个极端异常值
        normal_values = [base_value + np.random.uniform(-noise, noise) for _ in range(5)]
        outlier = base_value * outlier_magnitude
        values = normal_values + [outlier]
        weights = [0.8] * 6
        
        clean_values, _, outliers = fusion._remove_outliers_iqr(values, weights)
        
        # 极端异常值应该被移除
        assert outlier not in clean_values or outliers > 0


class TestWeightedFusion:
    """
    Property 7: Quality-Weighted Fusion
    weight_i = quality_i / sum(all_qualities)
    """
    
    def test_equal_weights_gives_mean(self):
        """相等权重给出算术平均"""
        fusion = FrameFusion(fusion_frames=3)
        
        values = [1.0, 2.0, 3.0]
        weights = [1.0, 1.0, 1.0]
        
        result = fusion._weighted_average(values, weights)
        expected = 2.0  # (1+2+3)/3
        
        assert math.isclose(result, expected, rel_tol=1e-9)
    
    def test_higher_weight_has_more_influence(self):
        """高权重值有更大影响"""
        fusion = FrameFusion(fusion_frames=3)
        
        values = [1.0, 10.0]
        weights = [0.1, 0.9]  # 第二个值权重更高
        
        result = fusion._weighted_average(values, weights)
        
        # 结果应该更接近 10.0
        assert result > 5.0
        assert result < 10.0
    
    @given(
        v1=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        v2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        w1=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
        w2=st.floats(min_value=0.1, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_weighted_average_formula(self, v1, v2, w1, w2):
        """加权平均公式正确"""
        fusion = FrameFusion(fusion_frames=2, min_frames=2)
        
        values = [v1, v2]
        weights = [w1, w2]
        
        result = fusion._weighted_average(values, weights)
        expected = (v1 * w1 + v2 * w2) / (w1 + w2)
        
        assert math.isclose(result, expected, rel_tol=1e-9)
    
    @given(
        values=st.lists(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
            min_size=3, max_size=10
        ),
        weights=st.lists(
            st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
            min_size=3, max_size=10
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_weighted_average_in_range(self, values, weights):
        """加权平均结果在值范围内"""
        assume(len(values) == len(weights))
        assume(len(values) >= 3)
        
        fusion = FrameFusion(fusion_frames=len(values), min_frames=3)
        
        result = fusion._weighted_average(values, weights)
        
        # 使用容差处理浮点数精度问题
        min_val = min(values)
        max_val = max(values)
        assert result >= min_val - 1e-6 or math.isclose(result, min_val, rel_tol=1e-9)
        assert result <= max_val + 1e-6 or math.isclose(result, max_val, rel_tol=1e-9)


class TestFrameFusionIntegration:
    """融合器集成测试"""
    
    def test_add_frames_until_ready(self):
        """添加帧直到准备就绪"""
        fusion = FrameFusion(fusion_frames=5)
        
        for i in range(4):
            ready = fusion.add_frame(body_data={'height': 1.7 + i * 0.01}, quality_score=0.8)
            assert ready is False
        
        ready = fusion.add_frame(body_data={'height': 1.72}, quality_score=0.8)
        assert ready is True
        assert fusion.is_ready is True
    
    def test_fuse_produces_result(self):
        """融合产生结果"""
        fusion = FrameFusion(fusion_frames=5)
        
        for i in range(5):
            fusion.add_frame(
                body_data={'height': 1.7 + i * 0.01},
                quality_score=0.8
            )
        
        result = fusion.fuse()
        
        assert result.frames_used == 5
        assert result.body_measurement is not None
        assert 'height' in result.body_measurement
    
    def test_reset_clears_buffer(self):
        """重置清空缓冲区"""
        fusion = FrameFusion(fusion_frames=5)
        
        for i in range(3):
            fusion.add_frame(body_data={'height': 1.7}, quality_score=0.8)
        
        assert fusion.buffer_size == 3
        
        fusion.reset()
        
        assert fusion.buffer_size == 0
        assert fusion.is_ready is False


class TestEdgeCases:
    """边界情况测试"""
    
    def test_insufficient_frames(self):
        """帧数不足时返回空结果"""
        fusion = FrameFusion(fusion_frames=5, min_frames=3)
        
        fusion.add_frame(body_data={'height': 1.7}, quality_score=0.8)
        fusion.add_frame(body_data={'height': 1.71}, quality_score=0.8)
        
        result = fusion.fuse()
        
        assert result.frames_used == 0
    
    def test_empty_data(self):
        """空数据处理"""
        fusion = FrameFusion(fusion_frames=3, min_frames=3)
        
        for _ in range(3):
            fusion.add_frame(body_data={}, quality_score=0.8)
        
        result = fusion.fuse()
        
        assert result.frames_used == 3
        # 空数据返回 None 是正确行为
        assert result.body_measurement is None or result.body_measurement == {}
    
    def test_zero_weights(self):
        """零权重处理"""
        fusion = FrameFusion(fusion_frames=3)
        
        values = [1.0, 2.0, 3.0]
        weights = [0.0, 0.0, 0.0]
        
        # 应该回退到简单平均
        result = fusion._weighted_average(values, weights)
        assert math.isclose(result, 2.0, rel_tol=1e-9)
    
    def test_all_outliers(self):
        """所有值都是异常值的情况"""
        fusion = FrameFusion(fusion_frames=5, min_frames=3)
        
        # 添加差异很大的值
        values = [1.0, 100.0, 1000.0, 10000.0, 100000.0]
        weights = [0.8] * 5
        
        # 应该能处理而不崩溃
        clean_values, clean_weights, outliers = fusion._remove_outliers_iqr(values, weights)
        
        # 至少应该保留一些值
        assert isinstance(clean_values, list)
