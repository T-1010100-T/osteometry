"""
数据验证工具
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ValidationError(Exception):
    """验证错误"""
    pass


def validate_image(image: np.ndarray, expected_channels: int = 3) -> bool:
    """
    验证图像数据
    
    Args:
        image: 图像数组
        expected_channels: 期望的通道数
    
    Returns:
        是否有效
    
    Raises:
        ValidationError: 验证失败
    """
    if image is None:
        raise ValidationError("图像为空")
    
    if not isinstance(image, np.ndarray):
        raise ValidationError(f"图像类型错误: {type(image)}")
    
    if len(image.shape) != 3:
        raise ValidationError(f"图像维度错误: {image.shape}")
    
    if image.shape[2] != expected_channels:
        raise ValidationError(f"通道数错误: {image.shape[2]} != {expected_channels}")
    
    return True


def validate_depth(depth: np.ndarray, min_val: float = 0.0, max_val: float = 10.0) -> bool:
    """
    验证深度图数据
    
    Args:
        depth: 深度图数组
        min_val: 最小有效值
        max_val: 最大有效值
    
    Returns:
        是否有效
    """
    if depth is None:
        raise ValidationError("深度图为空")
    
    if not isinstance(depth, np.ndarray):
        raise ValidationError(f"深度图类型错误: {type(depth)}")
    
    if len(depth.shape) != 2:
        raise ValidationError(f"深度图维度错误: {depth.shape}")
    
    return True


def validate_landmarks(landmarks: List, min_count: int = 33) -> bool:
    """
    验证关键点数据
    
    Args:
        landmarks: 关键点列表
        min_count: 最小数量
    
    Returns:
        是否有效
    """
    if landmarks is None:
        raise ValidationError("关键点为空")
    
    if len(landmarks) < min_count:
        raise ValidationError(f"关键点数量不足: {len(landmarks)} < {min_count}")
    
    return True


def validate_measurement(value: float, min_val: float, max_val: float, name: str = "测量值") -> bool:
    """
    验证测量值范围
    
    Args:
        value: 测量值
        min_val: 最小值
        max_val: 最大值
        name: 名称（用于错误信息）
    
    Returns:
        是否有效
    """
    if value < min_val or value > max_val:
        raise ValidationError(f"{name}超出范围: {value} 不在 [{min_val}, {max_val}]")
    
    return True


def validate_config(config: Dict, required_keys: List[str]) -> bool:
    """
    验证配置字典
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
    
    Returns:
        是否有效
    """
    if config is None:
        raise ValidationError("配置为空")
    
    for key in required_keys:
        if key not in config:
            raise ValidationError(f"缺少配置项: {key}")
    
    return True
