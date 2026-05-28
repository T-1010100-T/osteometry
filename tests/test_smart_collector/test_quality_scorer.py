"""
质量评分器测试

**Feature: smart-data-collector**
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.2, 7.3, 7.4**
"""
import pytest
import math
from hypothesis import given, strategies as st, settings, HealthCheck, assume

from src.core.quality_scorer import QualityScorer


# ============== Property Tests ==============

class TestQualityScoreFormula:
    """
    Property 4: Quality Score Formula
    score = confidence × 0.5 + stability × 0.4 + completeness × 0.1
    """
    
    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        stability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        completeness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_score_formula_correct(self, confidence: float, stability: float, completeness: float):
        """质量分数公式计算正确"""
        scorer = QualityScorer()
        result = scorer.calculate_score_direct(confidence, stability, completeness)
        
        expected = confidence * 0.5 + stability * 0.4 + completeness * 0.1
        assert math.isclose(result.score, expected, rel_tol=1e-9)
    
    @given(
        c_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        s_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        p_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        stability=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        completeness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_custom_weights_applied(
        self, c_weight, s_weight, p_weight, confidence, stability, completeness
    ):
        """自定义权重被正确应用"""
        scorer = QualityScorer(
            confidence_weight=c_weight,
            stability_weight=s_weight,
            completeness_weight=p_weight
        )
        result = scorer.calculate_score_direct(confidence, stability, completeness)
        
        expected = confidence * c_weight + stability * s_weight + completeness * p_weight
        assert math.isclose(result.score, expected, rel_tol=1e-9)


class TestQualityThresholds:
    """
    Property 5: Quality Threshold Recommendations
    score > 0.8 → auto_save, 0.6 ≤ score ≤ 0.8 → confirm, score < 0.6 → retry
    """
    
    @given(st.floats(min_value=0.801, max_value=1.0, allow_nan=False))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_high_score_auto_save(self, score: float):
        """高分数应推荐自动保存"""
        scorer = QualityScorer()
        # 设置所有分数相同以得到目标总分
        result = scorer.calculate_score_direct(score, score, score)
        assert result.recommendation == 'auto_save'
    
    @given(st.floats(min_value=0.0, max_value=0.599, allow_nan=False))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_low_score_retry(self, score: float):
        """低分数应推荐重试"""
        scorer = QualityScorer()
        result = scorer.calculate_score_direct(score, score, score)
        assert result.recommendation == 'retry'
    
    def test_medium_score_confirm(self):
        """中等分数应推荐确认"""
        scorer = QualityScorer()
        # 0.7 * (0.5 + 0.4 + 0.1) = 0.7，在 [0.6, 0.8] 范围内
        result = scorer.calculate_score_direct(0.7, 0.7, 0.7)
        assert result.recommendation == 'confirm'
    
    @given(
        auto_threshold=st.floats(min_value=0.5, max_value=0.95, allow_nan=False),
        confirm_threshold=st.floats(min_value=0.2, max_value=0.5, allow_nan=False)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_custom_thresholds(self, auto_threshold: float, confirm_threshold: float):
        """自定义阈值被正确应用"""
        assume(auto_threshold > confirm_threshold)
        
        scorer = QualityScorer(
            auto_save_threshold=auto_threshold,
            confirm_threshold=confirm_threshold
        )
        
        # 测试高于自动保存阈值
        high_score = auto_threshold + 0.01
        result = scorer.calculate_score_direct(high_score, high_score, high_score)
        assert result.recommendation == 'auto_save'
        
        # 测试低于确认阈值
        low_score = confirm_threshold - 0.01
        result = scorer.calculate_score_direct(low_score, low_score, low_score)
        assert result.recommendation == 'retry'


class TestAnatomicalValidation:
    """
    Property 10: Anatomical Validation
    """
    
    @given(
        height=st.floats(min_value=1.5, max_value=2.0, allow_nan=False),
        arm_ratio=st.floats(min_value=0.5, max_value=0.99, allow_nan=False)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_arm_shorter_than_height_valid(self, height: float, arm_ratio: float):
        """臂长小于身高时验证通过"""
        scorer = QualityScorer()
        arm_length = height * arm_ratio
        
        valid, issues = scorer.validate_anatomy_direct(
            height=height,
            left_arm=arm_length,
            right_arm=arm_length
        )
        
        # 臂长小于身高，应该通过这项检查
        arm_issues = [i for i in issues if "臂长超过身高" in i]
        assert len(arm_issues) == 0
    
    @given(
        height=st.floats(min_value=1.5, max_value=2.0, allow_nan=False),
        arm_ratio=st.floats(min_value=1.01, max_value=1.5, allow_nan=False)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_arm_longer_than_height_invalid(self, height: float, arm_ratio: float):
        """臂长大于等于身高时验证失败"""
        scorer = QualityScorer()
        arm_length = height * arm_ratio
        
        valid, issues = scorer.validate_anatomy_direct(
            height=height,
            left_arm=arm_length,
            right_arm=arm_length
        )
        
        assert valid is False
        assert any("臂长超过身高" in i for i in issues)
    
    @given(
        base_length=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        diff_ratio=st.floats(min_value=0.0, max_value=0.04, allow_nan=False)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_symmetric_limbs_valid(self, base_length: float, diff_ratio: float):
        """左右对称（差异<5%）时验证通过"""
        scorer = QualityScorer()
        left = base_length
        right = base_length * (1 + diff_ratio)
        
        valid, issues = scorer.validate_anatomy_direct(
            height=2.0,  # 足够高
            left_arm=left,
            right_arm=right
        )
        
        # 差异小于5%，应该通过对称性检查
        sym_issues = [i for i in issues if "差异过大" in i]
        assert len(sym_issues) == 0
    
    @given(
        base_length=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        diff_ratio=st.floats(min_value=0.20, max_value=0.5, allow_nan=False)  # 容差已放宽到15%，测试用20%以上
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_asymmetric_limbs_invalid(self, base_length: float, diff_ratio: float):
        """左右不对称（差异>15%）时验证失败"""
        scorer = QualityScorer()
        left = base_length
        right = base_length * (1 + diff_ratio)
        
        valid, issues = scorer.validate_anatomy_direct(
            height=2.0,
            left_arm=left,
            right_arm=right
        )
        
        assert any("差异过大" in i for i in issues)


class TestFingerProportions:
    """手指比例验证测试"""
    
    def test_normal_finger_proportions_valid(self):
        """正常手指比例验证通过"""
        scorer = QualityScorer()
        
        # 正常比例: middle > ring > index > pinky > thumb
        finger_lengths = {
            'thumb': 0.05,
            'index': 0.07,
            'middle': 0.08,
            'ring': 0.075,
            'pinky': 0.06
        }
        
        issues = scorer._validate_finger_proportions(finger_lengths)
        assert len(issues) == 0
    
    def test_abnormal_finger_proportions_invalid(self):
        """异常手指比例验证失败"""
        scorer = QualityScorer()
        
        # 异常比例: 小指比中指长
        finger_lengths = {
            'thumb': 0.05,
            'index': 0.07,
            'middle': 0.06,  # 中指太短
            'ring': 0.075,
            'pinky': 0.08   # 小指太长
        }
        
        issues = scorer._validate_finger_proportions(finger_lengths)
        assert len(issues) > 0


class TestEdgeCases:
    """边界情况测试"""
    
    def test_zero_scores(self):
        """零分数处理"""
        scorer = QualityScorer()
        result = scorer.calculate_score_direct(0.0, 0.0, 0.0)
        assert result.score == 0.0
        assert result.recommendation == 'retry'
    
    def test_perfect_scores(self):
        """满分处理"""
        scorer = QualityScorer()
        result = scorer.calculate_score_direct(1.0, 1.0, 1.0)
        assert result.score == 1.0
        assert result.recommendation == 'auto_save'
    
    def test_missing_data_issues_reported(self):
        """缺失数据时报告问题"""
        scorer = QualityScorer()
        result = scorer.calculate_score()
        assert len(result.issues) > 0
