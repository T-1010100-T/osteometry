"""
关键点稳定器配置模块

包含 One-Euro 滤波器和关键点稳定器的配置类

深度优化：
- 新增 measurement 预设：专为静态身高测量场景设计
- 更强的3D滤波参数，减少深度噪声引起的测量抖动
- 更严格的跳变检测阈值，过滤异常关键点
"""
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class OneEuroConfig:
    """
    One-Euro 滤波器配置
    
    One-Euro 滤波器是一种自适应低通滤波器：
    - 静止时：使用低截止频率，强平滑
    - 运动时：使用高截止频率，弱平滑（减少延迟）
    """
    
    mincutoff: float = 1.0
    beta: float = 0.05
    dcutoff: float = 1.0


@dataclass
class StabilizerConfig:
    """
    关键点稳定器配置
    """
    
    freq: float = 30.0
    
    filter_2d: OneEuroConfig = field(default_factory=lambda: OneEuroConfig(
        mincutoff=1.0,
        beta=0.05,
        dcutoff=1.0
    ))
    
    filter_3d: OneEuroConfig = field(default_factory=lambda: OneEuroConfig(
        mincutoff=0.5,
        beta=0.1,
        dcutoff=1.0
    ))
    
    low_confidence_threshold: float = 0.3
    medium_confidence_threshold: float = 0.6
    
    history_size: int = 5
    
    max_jump_2d: float = 50.0
    max_jump_3d: float = 0.1
    
    enable_2d_stabilization: bool = True
    enable_3d_stabilization: bool = True


STABILIZER_PRESETS: Dict[str, StabilizerConfig] = {
    'default': StabilizerConfig(),
    
    'smooth': StabilizerConfig(
        filter_2d=OneEuroConfig(mincutoff=0.5, beta=0.02, dcutoff=1.0),
        filter_3d=OneEuroConfig(mincutoff=0.3, beta=0.05, dcutoff=1.0),
    ),
    
    'responsive': StabilizerConfig(
        filter_2d=OneEuroConfig(mincutoff=2.0, beta=0.1, dcutoff=1.0),
        filter_3d=OneEuroConfig(mincutoff=1.0, beta=0.2, dcutoff=1.0),
    ),
    
    'minimal': StabilizerConfig(
        filter_2d=OneEuroConfig(mincutoff=5.0, beta=0.5, dcutoff=1.0),
        filter_3d=OneEuroConfig(mincutoff=3.0, beta=0.5, dcutoff=1.0),
    ),
    
    'measurement': StabilizerConfig(
        freq=30.0,
        filter_2d=OneEuroConfig(mincutoff=0.4, beta=0.01, dcutoff=0.8),
        filter_3d=OneEuroConfig(mincutoff=0.8, beta=0.1, dcutoff=1.0),
        low_confidence_threshold=0.4,
        medium_confidence_threshold=0.65,
        history_size=8,
        max_jump_2d=30.0,
        max_jump_3d=0.05,
        enable_2d_stabilization=True,
        enable_3d_stabilization=True,
    ),
}


def get_stabilizer_preset(name: str) -> StabilizerConfig:
    """
    获取预设配置
    
    Args:
        name: 预设名称 (default, smooth, responsive, minimal, measurement)
    
    Returns:
        StabilizerConfig 实例
    """
    return STABILIZER_PRESETS.get(name, STABILIZER_PRESETS['default'])
