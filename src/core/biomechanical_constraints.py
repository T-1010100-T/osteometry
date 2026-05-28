"""
生物力学约束模块

提供人体比例验证、对称性检查和置信度评分
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ConstraintsConfig:
    """
    生物力学约束配置
    """
    
    # 比例约束
    enable_proportion_check: bool = True
    proportion_tolerance: float = 0.1  # 允许超出范围 10%
    
    # 对称性约束
    enable_symmetry_check: bool = True
    symmetry_threshold: float = 0.08  # 8% 差异阈值

    # 身高范围（米）
    min_height: float = 1.4
    max_height: float = 2.2


# 人体比例约束（骨骼长度 / 身高）
# 基于人体测量学数据
PROPORTION_CONSTRAINTS: Dict[str, Tuple[float, float, float]] = {
    # 骨骼名称: (最小比例, 典型比例, 最大比例)
    
    # 上肢
    'upper_arm': (0.15, 0.175, 0.20),      # 上臂
    'forearm': (0.13, 0.15, 0.17),          # 前臂
    'hand': (0.09, 0.105, 0.12),            # 手
    
    # 下肢
    'thigh': (0.23, 0.26, 0.28),            # 大腿
    'shin': (0.22, 0.245, 0.28),            # 小腿
    'foot': (0.13, 0.15, 0.17),             # 脚
    
    # 躯干
    'torso': (0.28, 0.30, 0.33),            # 躯干（肩到髋）
    'spine_lower': (0.08, 0.10, 0.12),      # 下脊柱
    'spine_upper': (0.10, 0.12, 0.14),      # 上脊柱
    'shoulder_width': (0.20, 0.23, 0.26),   # 肩宽
    'hip_width': (0.14, 0.16, 0.18),        # 髋宽
}

# 骨骼名称映射（中文 -> 英文类别）
BONE_NAME_MAPPING: Dict[str, str] = {
    # 左上肢
    '左肩到左肘': 'upper_arm',
    '左肘到左腕': 'forearm',
    '左腕到左手': 'hand',
    
    # 右上肢
    '右肩到右肘': 'upper_arm',
    '右肘到右腕': 'forearm',
    '右腕到右手': 'hand',
    
    # 左下肢
    '左髋到左膝': 'thigh',
    '左膝到左踝': 'shin',
    '左踝到左脚': 'foot',
    
    # 右下肢
    '右髋到右膝': 'thigh',
    '右膝到右踝': 'shin',
    '右踝到右脚': 'foot',
    
    # 躯干
    '脊柱底部到中部': 'spine_lower',
    '脊柱中部到肩部': 'spine_upper',
    '脊柱肩部到左肩': 'shoulder_width',
    '脊柱肩部到右肩': 'shoulder_width',
    '脊柱底部到左髋': 'hip_width',
    '脊柱底部到右髋': 'hip_width',
}

# 对称骨骼对
SYMMETRIC_BONE_PAIRS: List[Tuple[str, str, str]] = [
    # (左侧骨骼, 右侧骨骼, 类别名称)
    ('左肩到左肘', '右肩到右肘', '上臂'),
    ('左肘到左腕', '右肘到右腕', '前臂'),
    ('左腕到左手', '右腕到右手', '手'),
    ('左髋到左膝', '右髋到右膝', '大腿'),
    ('左膝到左踝', '右膝到右踝', '小腿'),
    ('左踝到左脚', '右踝到右脚', '脚'),
    ('脊柱肩部到左肩', '脊柱肩部到右肩', '肩宽'),
    ('脊柱底部到左髋', '脊柱底部到右髋', '髋宽'),
]


@dataclass
class ProportionResult:
    """比例验证结果"""
    bone_name: str
    value_cm: float
    ratio: float
    min_ratio: float
    max_ratio: float
    valid: bool
    message: str = ""


@dataclass
class SymmetryResult:
    """对称性检查结果"""
    category: str
    left_name: str
    right_name: str
    left_value: float
    right_value: float
    difference: float
    ratio: float
    symmetric: bool
    message: str = ""


@dataclass
class ConstraintsResult:
    """约束检查结果"""
    estimated_height_cm: float
    proportion_results: Dict[str, ProportionResult]
    symmetry_results: Dict[str, SymmetryResult]
    confidence_score: float
    warnings: List[str]
    is_valid: bool


class BiomechanicalConstraints:
    """
    生物力学约束系统
    
    功能：
    - 从骨骼长度估计身高
    - 验证骨骼比例是否在合理范围
    - 检查左右对称性
    - 计算测量置信度评分
    """
    
    def __init__(self, config: Optional[ConstraintsConfig] = None):
        """
        初始化
        
        Args:
            config: 约束配置
        """
        self.config = config or ConstraintsConfig()
    
    def estimate_height(self, bone_lengths_cm: Dict[str, Optional[float]]) -> float:
        """
        从骨骼长度估计身高
        
        使用多种方法估计身高，取加权平均：
        1. 下肢法：(大腿 + 小腿) * 2 + 躯干
        2. 上肢法：臂展 ≈ 身高
        3. 躯干法：躯干 * 3.3
        
        Args:
            bone_lengths_cm: 骨骼长度字典（厘米）
        
        Returns:
            估计身高（厘米）
        """
        estimates = []
        weights = []
        
        # 方法1：下肢法
        left_leg = self._sum_bones(bone_lengths_cm, ['左髋到左膝', '左膝到左踝'])
        right_leg = self._sum_bones(bone_lengths_cm, ['右髋到右膝', '右膝到右踝'])
        torso = self._sum_bones(bone_lengths_cm, ['脊柱底部到中部', '脊柱中部到肩部', '脊柱肩部到头部'])
        
        if left_leg and right_leg and torso:
            avg_leg = (left_leg + right_leg) / 2
            height_leg = avg_leg * 2 + torso
            estimates.append(height_leg)
            weights.append(1.0)
        elif left_leg and torso:
            height_leg = left_leg * 2 + torso
            estimates.append(height_leg)
            weights.append(0.7)
        elif right_leg and torso:
            height_leg = right_leg * 2 + torso
            estimates.append(height_leg)
            weights.append(0.7)
        
        # 方法2：上肢法（臂展 ≈ 身高）
        left_arm = self._sum_bones(bone_lengths_cm, ['脊柱肩部到左肩', '左肩到左肘', '左肘到左腕', '左腕到左手'])
        right_arm = self._sum_bones(bone_lengths_cm, ['脊柱肩部到右肩', '右肩到右肘', '右肘到右腕', '右腕到右手'])
        
        if left_arm and right_arm:
            arm_span = left_arm + right_arm
            estimates.append(arm_span)
            weights.append(0.8)
        
        # 方法3：躯干法
        if torso:
            height_torso = torso * 3.3
            estimates.append(height_torso)
            weights.append(0.5)
        
        # 加权平均
        if not estimates:
            return 170.0  # 默认身高
        
        total_weight = sum(weights)
        weighted_sum = sum(e * w for e, w in zip(estimates, weights))
        estimated = weighted_sum / total_weight
        
        # 限制在合理范围
        min_h = self.config.min_height * 100
        max_h = self.config.max_height * 100
        return max(min_h, min(max_h, estimated))
    
    def _sum_bones(
        self,
        bone_lengths_cm: Dict[str, Optional[float]],
        bone_names: List[str]
    ) -> Optional[float]:
        """求骨骼长度之和"""
        total = 0.0
        for name in bone_names:
            value = bone_lengths_cm.get(name)
            if value is None or value <= 0:
                return None
            total += value
        return total
    
    def validate_proportions(
        self,
        bone_lengths_cm: Dict[str, Optional[float]],
        height_cm: float
    ) -> Dict[str, ProportionResult]:
        """
        验证骨骼比例
        
        Args:
            bone_lengths_cm: 骨骼长度字典（厘米）
            height_cm: 身高（厘米）
        
        Returns:
            各骨骼的比例验证结果
        """
        results = {}
        
        if height_cm <= 0:
            return results
        
        tolerance = self.config.proportion_tolerance
        
        for bone_name, value in bone_lengths_cm.items():
            if value is None or value <= 0:
                continue
            
            category = BONE_NAME_MAPPING.get(bone_name)
            if category is None or category not in PROPORTION_CONSTRAINTS:
                continue
            
            min_ratio, typical_ratio, max_ratio = PROPORTION_CONSTRAINTS[category]
            
            # 应用容差
            min_with_tolerance = min_ratio * (1 - tolerance)
            max_with_tolerance = max_ratio * (1 + tolerance)
            
            ratio = value / height_cm
            valid = min_with_tolerance <= ratio <= max_with_tolerance
            
            message = ""
            if ratio < min_with_tolerance:
                message = f"偏短 ({ratio:.3f} < {min_ratio:.3f})"
            elif ratio > max_with_tolerance:
                message = f"偏长 ({ratio:.3f} > {max_ratio:.3f})"
            
            results[bone_name] = ProportionResult(
                bone_name=bone_name,
                value_cm=value,
                ratio=ratio,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                valid=valid,
                message=message
            )
        
        return results

    def check_symmetry(
        self,
        bone_lengths_cm: Dict[str, Optional[float]]
    ) -> Dict[str, SymmetryResult]:
        """
        检查左右对称性
        
        Args:
            bone_lengths_cm: 骨骼长度字典（厘米）
        
        Returns:
            各对称骨骼对的检查结果
        """
        results = {}
        threshold = self.config.symmetry_threshold
        
        for left_name, right_name, category in SYMMETRIC_BONE_PAIRS:
            left_value = bone_lengths_cm.get(left_name)
            right_value = bone_lengths_cm.get(right_name)
            
            if left_value is None or right_value is None:
                continue
            if left_value <= 0 or right_value <= 0:
                continue
            
            # 计算差异
            difference = abs(left_value - right_value)
            avg_value = (left_value + right_value) / 2
            ratio = min(left_value, right_value) / max(left_value, right_value)
            
            # 判断是否对称
            relative_diff = difference / avg_value
            symmetric = relative_diff <= threshold
            
            message = ""
            if not symmetric:
                if left_value > right_value:
                    message = f"左侧偏长 {difference:.1f}cm ({relative_diff*100:.1f}%)"
                else:
                    message = f"右侧偏长 {difference:.1f}cm ({relative_diff*100:.1f}%)"
            
            results[category] = SymmetryResult(
                category=category,
                left_name=left_name,
                right_name=right_name,
                left_value=left_value,
                right_value=right_value,
                difference=difference,
                ratio=ratio,
                symmetric=symmetric,
                message=message
            )
        
        return results
    
    def get_confidence_score(
        self,
        bone_lengths_cm: Dict[str, Optional[float]],
        height_cm: float
    ) -> float:
        """
        计算测量置信度评分
        
        评分基于：
        1. 比例合理性（40%）
        2. 对称性（30%）
        3. 数据完整性（30%）
        
        Args:
            bone_lengths_cm: 骨骼长度字典（厘米）
            height_cm: 身高（厘米）
        
        Returns:
            置信度 (0-1)
        """
        scores = []
        weights = []
        
        # 1. 比例合理性评分
        if self.config.enable_proportion_check:
            proportion_results = self.validate_proportions(bone_lengths_cm, height_cm)
            if proportion_results:
                valid_count = sum(1 for r in proportion_results.values() if r.valid)
                total_count = len(proportion_results)
                proportion_score = valid_count / total_count if total_count > 0 else 0.5
                scores.append(proportion_score)
                weights.append(0.4)
        
        # 2. 对称性评分
        if self.config.enable_symmetry_check:
            symmetry_results = self.check_symmetry(bone_lengths_cm)
            if symmetry_results:
                symmetric_count = sum(1 for r in symmetry_results.values() if r.symmetric)
                total_count = len(symmetry_results)
                symmetry_score = symmetric_count / total_count if total_count > 0 else 0.5
                scores.append(symmetry_score)
                weights.append(0.3)
        
        # 3. 数据完整性评分
        expected_bones = [
            '身高', '左肩到左肘', '左肘到左腕', '右肩到右肘', '右肘到右腕',
            '左髋到左膝', '左膝到左踝', '右髋到右膝', '右膝到右踝'
        ]
        valid_count = sum(1 for name in expected_bones 
                        if bone_lengths_cm.get(name) is not None 
                        and bone_lengths_cm.get(name) > 0)
        completeness_score = valid_count / len(expected_bones)
        scores.append(completeness_score)
        weights.append(0.3)
        
        # 加权平均
        if not scores:
            return 0.5
        
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return weighted_sum / total_weight
    
    def validate(
        self,
        bone_lengths_cm: Dict[str, Optional[float]],
        provided_height_cm: Optional[float] = None
    ) -> ConstraintsResult:
        """
        执行完整的约束验证
        
        Args:
            bone_lengths_cm: 骨骼长度字典（厘米）
            provided_height_cm: 提供的身高（厘米），如果为 None 则自动估计
        
        Returns:
            约束检查结果
        """
        warnings = []
        
        # 估计或使用提供的身高
        if provided_height_cm is not None and provided_height_cm > 0:
            height_cm = provided_height_cm
        else:
            height_cm = self.estimate_height(bone_lengths_cm)
        
        # 比例验证
        proportion_results = {}
        if self.config.enable_proportion_check:
            proportion_results = self.validate_proportions(bone_lengths_cm, height_cm)
            for name, result in proportion_results.items():
                if not result.valid:
                    warnings.append(f"比例异常: {name} {result.message}")
        
        # 对称性检查
        symmetry_results = {}
        if self.config.enable_symmetry_check:
            symmetry_results = self.check_symmetry(bone_lengths_cm)
            for category, result in symmetry_results.items():
                if not result.symmetric:
                    warnings.append(f"不对称: {category} {result.message}")
        
        # 置信度评分
        confidence = self.get_confidence_score(bone_lengths_cm, height_cm)
        
        # 判断整体有效性
        proportion_valid = all(r.valid for r in proportion_results.values()) if proportion_results else True
        symmetry_valid = all(r.symmetric for r in symmetry_results.values()) if symmetry_results else True
        is_valid = proportion_valid and symmetry_valid and confidence >= 0.6
        
        return ConstraintsResult(
            estimated_height_cm=height_cm,
            proportion_results=proportion_results,
            symmetry_results=symmetry_results,
            confidence_score=confidence,
            warnings=warnings,
            is_valid=is_valid
        )
    
    def apply_constraints(
        self,
        bone_lengths_cm: Dict[str, Optional[float]],
        height_cm: float
    ) -> Dict[str, Optional[float]]:
        """
        应用约束校正（已废弃，保留接口兼容性）

        校正功能已移除，直接返回原始数据

        Args:
            bone_lengths_cm: 骨骼长度字典（厘米）
            height_cm: 身高（厘米）

        Returns:
            原始骨骼长度字典（无校正）
        """
        return bone_lengths_cm
