"""
数学工具函数
向量运算、距离计算等
"""
from typing import List, Tuple, Union
import numpy as np


def euclidean_distance(p1: Union[Tuple, np.ndarray], p2: Union[Tuple, np.ndarray]) -> float:
    """
    计算两点间的欧几里得距离
    
    Args:
        p1, p2: 点坐标 (x, y, z) 或 numpy 数组
    
    Returns:
        距离
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    return float(np.linalg.norm(p1 - p2))


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两向量夹角
    
    Args:
        v1, v2: 向量
    
    Returns:
        角度（度）
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return float(np.degrees(np.arccos(cos_angle)))


def moving_average(values: List[float], window: int) -> float:
    """
    滑动平均
    
    Args:
        values: 数值列表
        window: 窗口大小
    
    Returns:
        平均值
    """
    if not values:
        return 0.0
    
    window = min(window, len(values))
    return float(np.mean(values[-window:]))


def remove_outliers(values: List[float], threshold: float = 2.0) -> List[float]:
    """
    移除异常值（基于标准差）
    
    Args:
        values: 数值列表
        threshold: 标准差倍数阈值
    
    Returns:
        过滤后的列表
    """
    if len(values) < 3:
        return values
    
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)
    
    if std == 0:
        return values
    
    return [v for v in values if abs(v - mean) <= threshold * std]


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """归一化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """
    计算点到线段的距离
    
    Args:
        point: 点坐标
        line_start, line_end: 线段端点
    
    Returns:
        距离
    """
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return float(np.linalg.norm(point_vec))
    
    line_unit = line_vec / line_len
    proj_length = np.dot(point_vec, line_unit)
    proj_length = np.clip(proj_length, 0, line_len)
    
    proj_point = line_start + proj_length * line_unit
    return float(np.linalg.norm(point - proj_point))
