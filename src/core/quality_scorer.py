"""
质量评分器

综合置信度、稳定性、完整性计算数据质量分数

**Feature: smart-data-collector**
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.2, 7.3, 7.4**
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any
import numpy as np

from .smart_collector_types import StabilityResult, QualityResult
from .hand_result import HolisticResult
from ..utils.logger import get_logger

# 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from ..measurement.measurement_engine import MeasurementResult
    from ..measurement.hand_measurement import HandMeasurementResult

logger = get_logger(__name__)


class QualityScorer:
    """
    质量评分器
    
    综合置信度、稳定性、完整性计算数据质量分数
    并提供基于阈值的保存建议
    
    Example:
        >>> scorer = QualityScorer()
        >>> result = scorer.calculate_score(holistic, stability, measurement)
        >>> if result.recommendation == 'auto_save':
        ...     save_data()
    """
    
    # 默认权重
    DEFAULT_CONFIDENCE_WEIGHT = 0.5
    DEFAULT_STABILITY_WEIGHT = 0.4
    DEFAULT_COMPLETENESS_WEIGHT = 0.1
    
    # 默认阈值
    DEFAULT_AUTO_SAVE_THRESHOLD = 0.8
    DEFAULT_CONFIRM_THRESHOLD = 0.6
    
    # 解剖学验证参数
    SYMMETRY_TOLERANCE = 0.15  # 15% 左右对称容差（放宽以适应深度误差）
    
    # 骨骼长度比例约束（相对于身高）
    BONE_RATIO_CONSTRAINTS = {
        'thigh': (0.20, 0.32),      # 大腿长度占身高比例
        'calf': (0.18, 0.30),       # 小腿长度占身高比例
        'upper_arm': (0.14, 0.22),  # 上臂长度占身高比例
        'forearm': (0.12, 0.20),    # 前臂长度占身高比例
    }
    
    def __init__(
        self,
        confidence_weight: float = DEFAULT_CONFIDENCE_WEIGHT,
        stability_weight: float = DEFAULT_STABILITY_WEIGHT,
        completeness_weight: float = DEFAULT_COMPLETENESS_WEIGHT,
        auto_save_threshold: float = DEFAULT_AUTO_SAVE_THRESHOLD,
        confirm_threshold: float = DEFAULT_CONFIRM_THRESHOLD,
        min_visibility: float = 0.5
    ):
        """
        初始化质量评分器
        
        Args:
            confidence_weight: 置信度权重
            stability_weight: 稳定性权重
            completeness_weight: 完整性权重
            auto_save_threshold: 自动保存阈值
            confirm_threshold: 需确认阈值
            min_visibility: 最小可见性阈值
        """
        self.confidence_weight = confidence_weight
        self.stability_weight = stability_weight
        self.completeness_weight = completeness_weight
        self.auto_save_threshold = auto_save_threshold
        self.confirm_threshold = confirm_threshold
        self.min_visibility = min_visibility
        
        logger.debug(f"QualityScorer 初始化: weights=({confidence_weight}, {stability_weight}, {completeness_weight})")

    
    def calculate_score(
        self,
        holistic_result: Optional[HolisticResult] = None,
        stability: Optional[StabilityResult] = None,
        body_measurement: Optional["MeasurementResult"] = None,
        hand_measurement: Optional["HandMeasurementResult"] = None,
        confidence: Optional[float] = None,
        stability_score: Optional[float] = None,
        completeness: Optional[float] = None
    ) -> QualityResult:
        """
        计算综合质量分数
        
        公式: score = confidence × 0.5 + stability × 0.4 + completeness × 0.1
        
        Args:
            holistic_result: Holistic 检测结果
            stability: 稳定性检测结果
            body_measurement: 身体测量结果
            hand_measurement: 手部测量结果
            confidence: 直接提供置信度分数 (0-1)
            stability_score: 直接提供稳定性分数 (0-1)
            completeness: 直接提供完整性分数 (0-1)
        
        Returns:
            QualityResult: 质量评分结果
        """
        issues = []
        
        # 计算置信度分数
        if confidence is not None:
            conf_score = confidence
        elif holistic_result is not None:
            conf_score = self._calculate_confidence_score(holistic_result)
        elif body_measurement is not None:
            conf_score = body_measurement.overall_confidence
        else:
            conf_score = 0.0
            issues.append("无法计算置信度")
        
        # 计算稳定性分数
        if stability_score is not None:
            stab_score = stability_score
        elif stability is not None:
            stab_score = stability.progress
        else:
            stab_score = 0.0
            issues.append("无稳定性数据")
        
        # 计算完整性分数
        if completeness is not None:
            comp_score = completeness
        elif holistic_result is not None:
            comp_score = self._calculate_completeness_score(holistic_result)
        else:
            comp_score = 0.0
            issues.append("无法计算完整性")
        
        # 计算综合分数
        total_score = (
            conf_score * self.confidence_weight +
            stab_score * self.stability_weight +
            comp_score * self.completeness_weight
        )
        
        # 确定建议
        recommendation = self._get_recommendation(total_score)
        
        # 解剖学验证
        anatomy_valid = True
        anatomy_issues = []
        if body_measurement is not None:
            anatomy_valid, anatomy_issues = self.validate_anatomy(
                body_measurement, hand_measurement
            )
        
        return QualityResult(
            score=total_score,
            confidence_score=conf_score,
            stability_score=stab_score,
            completeness_score=comp_score,
            recommendation=recommendation,
            issues=issues,
            anatomy_valid=anatomy_valid,
            anatomy_issues=anatomy_issues
        )
    
    def calculate_score_direct(
        self,
        confidence: float,
        stability: float,
        completeness: float
    ) -> QualityResult:
        """
        直接使用分数计算（用于测试）
        
        Args:
            confidence: 置信度分数 (0-1)
            stability: 稳定性分数 (0-1)
            completeness: 完整性分数 (0-1)
        
        Returns:
            QualityResult: 质量评分结果
        """
        return self.calculate_score(
            confidence=confidence,
            stability_score=stability,
            completeness=completeness
        )
    
    def _calculate_confidence_score(self, holistic_result: HolisticResult) -> float:
        """计算置信度分数"""
        scores = []
        
        # 身体置信度 - 从关键点可见性计算
        if holistic_result.pose.detected and holistic_result.pose.landmarks:
            visible_count = sum(1 for lm in holistic_result.pose.landmarks if lm.visibility >= self.min_visibility)
            pose_conf = visible_count / len(holistic_result.pose.landmarks) if holistic_result.pose.landmarks else 0
            scores.append(pose_conf)
        
        # 手部置信度
        if holistic_result.left_hand.detected:
            scores.append(holistic_result.left_hand.confidence)
        if holistic_result.right_hand.detected:
            scores.append(holistic_result.right_hand.confidence)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_completeness_score(self, holistic_result: HolisticResult) -> float:
        """计算完整性分数"""
        total_expected = 0
        total_visible = 0
        
        # 身体关键点
        if holistic_result.pose.landmarks:
            total_expected += len(holistic_result.pose.landmarks)
            total_visible += sum(
                1 for lm in holistic_result.pose.landmarks
                if lm.visibility >= self.min_visibility
            )
        
        # 左手关键点
        if holistic_result.left_hand.landmarks:
            total_expected += len(holistic_result.left_hand.landmarks)
            total_visible += len(holistic_result.left_hand.landmarks)  # 手部无visibility
        
        # 右手关键点
        if holistic_result.right_hand.landmarks:
            total_expected += len(holistic_result.right_hand.landmarks)
            total_visible += len(holistic_result.right_hand.landmarks)
        
        return total_visible / total_expected if total_expected > 0 else 0.0
    
    def _get_recommendation(self, score: float) -> str:
        """根据分数获取建议"""
        if score > self.auto_save_threshold:
            return 'auto_save'
        elif score >= self.confirm_threshold:
            return 'confirm'
        else:
            return 'retry'
    
    def validate_anatomy(
        self,
        body_measurement: "MeasurementResult",
        hand_measurement: Optional["HandMeasurementResult"] = None
    ) -> Tuple[bool, List[str]]:
        """
        解剖学验证
        
        Args:
            body_measurement: 身体测量结果
            hand_measurement: 手部测量结果
        
        Returns:
            (是否有效, 问题列表)
        """
        issues = []
        
        # 1. 臂长 < 身高
        if body_measurement.height > 0:
            if body_measurement.left_arm_length >= body_measurement.height:
                issues.append(f"左臂长({body_measurement.left_arm_length:.2f}m)超过身高({body_measurement.height:.2f}m)")
            if body_measurement.right_arm_length >= body_measurement.height:
                issues.append(f"右臂长({body_measurement.right_arm_length:.2f}m)超过身高({body_measurement.height:.2f}m)")
        
        # 2. 左右对称性检查
        if body_measurement.left_arm_length > 0 and body_measurement.right_arm_length > 0:
            avg_arm = (body_measurement.left_arm_length + body_measurement.right_arm_length) / 2
            diff = abs(body_measurement.left_arm_length - body_measurement.right_arm_length)
            if diff / avg_arm > self.SYMMETRY_TOLERANCE:
                issues.append(f"左右臂长差异过大: {diff/avg_arm*100:.1f}%")
        
        if body_measurement.left_leg_length > 0 and body_measurement.right_leg_length > 0:
            avg_leg = (body_measurement.left_leg_length + body_measurement.right_leg_length) / 2
            diff = abs(body_measurement.left_leg_length - body_measurement.right_leg_length)
            if diff / avg_leg > self.SYMMETRY_TOLERANCE:
                issues.append(f"左右腿长差异过大: {diff/avg_leg*100:.1f}%")
        
        # 3. 骨骼长度比例约束验证
        if body_measurement.height > 0:
            bone_issues = self._validate_bone_ratios(body_measurement)
            issues.extend(bone_issues)
        
        # 4. 手指比例验证
        if hand_measurement and hand_measurement.finger_lengths:
            finger_issues = self._validate_finger_proportions(hand_measurement.finger_lengths)
            issues.extend(finger_issues)
        
        return len(issues) == 0, issues
    
    def _validate_bone_ratios(self, body_measurement: "MeasurementResult") -> List[str]:
        """
        验证骨骼长度比例是否在合理范围内
        
        Args:
            body_measurement: 身体测量结果
        
        Returns:
            问题列表
        """
        issues = []
        height = body_measurement.height
        
        if height <= 0:
            return issues
        
        # 检查大腿长度
        for side, thigh_len in [('左', body_measurement.left_thigh), ('右', body_measurement.right_thigh)]:
            if thigh_len > 0:
                ratio = thigh_len / height
                min_r, max_r = self.BONE_RATIO_CONSTRAINTS['thigh']
                if ratio < min_r or ratio > max_r:
                    issues.append(f"{side}大腿比例异常: {ratio*100:.1f}% (正常: {min_r*100:.0f}-{max_r*100:.0f}%)")
        
        # 检查小腿长度
        for side, calf_len in [('左', body_measurement.left_calf), ('右', body_measurement.right_calf)]:
            if calf_len > 0:
                ratio = calf_len / height
                min_r, max_r = self.BONE_RATIO_CONSTRAINTS['calf']
                if ratio < min_r or ratio > max_r:
                    issues.append(f"{side}小腿比例异常: {ratio*100:.1f}% (正常: {min_r*100:.0f}-{max_r*100:.0f}%)")
        
        # 检查上臂长度
        for side, upper_arm in [('左', body_measurement.left_upper_arm), ('右', body_measurement.right_upper_arm)]:
            if upper_arm > 0:
                ratio = upper_arm / height
                min_r, max_r = self.BONE_RATIO_CONSTRAINTS['upper_arm']
                if ratio < min_r or ratio > max_r:
                    issues.append(f"{side}上臂比例异常: {ratio*100:.1f}% (正常: {min_r*100:.0f}-{max_r*100:.0f}%)")
        
        # 检查前臂长度
        for side, forearm in [('左', body_measurement.left_forearm), ('右', body_measurement.right_forearm)]:
            if forearm > 0:
                ratio = forearm / height
                min_r, max_r = self.BONE_RATIO_CONSTRAINTS['forearm']
                if ratio < min_r or ratio > max_r:
                    issues.append(f"{side}前臂比例异常: {ratio*100:.1f}% (正常: {min_r*100:.0f}-{max_r*100:.0f}%)")
        
        return issues
    
    def _validate_finger_proportions(self, finger_lengths: Dict[str, float]) -> List[str]:
        """
        验证手指比例
        
        正常比例: middle > ring > index > pinky > thumb (长度)
        """
        issues = []
        
        # 获取各手指长度
        thumb = finger_lengths.get('thumb', 0)
        index = finger_lengths.get('index', 0)
        middle = finger_lengths.get('middle', 0)
        ring = finger_lengths.get('ring', 0)
        pinky = finger_lengths.get('pinky', 0)
        
        # 检查是否有足够数据
        if not all([thumb, index, middle, ring, pinky]):
            return []  # 数据不完整，跳过验证
        
        # 中指应该最长
        if middle < ring or middle < index:
            issues.append("中指应为最长手指")
        
        # 小指应该最短（除拇指外）
        if pinky > ring or pinky > index:
            issues.append("小指长度异常")
        
        return issues
    
    def validate_anatomy_direct(
        self,
        height: float,
        left_arm: float,
        right_arm: float,
        left_leg: float = 0,
        right_leg: float = 0
    ) -> Tuple[bool, List[str]]:
        """
        直接验证解剖学参数（用于测试）
        """
        issues = []
        
        # 臂长 < 身高
        if height > 0:
            if left_arm >= height:
                issues.append("左臂长超过身高")
            if right_arm >= height:
                issues.append("右臂长超过身高")
        
        # 左右对称性
        if left_arm > 0 and right_arm > 0:
            avg = (left_arm + right_arm) / 2
            diff = abs(left_arm - right_arm)
            if diff / avg > self.SYMMETRY_TOLERANCE:
                issues.append("左右臂长差异过大")
        
        if left_leg > 0 and right_leg > 0:
            avg = (left_leg + right_leg) / 2
            diff = abs(left_leg - right_leg)
            if diff / avg > self.SYMMETRY_TOLERANCE:
                issues.append("左右腿长差异过大")
        
        return len(issues) == 0, issues
