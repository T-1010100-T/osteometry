"""
深度处理配置模块

包含滤镜链、深度处理器和采样器的配置类
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict

import yaml


def _load_yaml_config(config_path: str) -> Optional[dict]:
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


@dataclass
class FilterChainConfig:
    """
    RealSense 滤镜链配置
    
    滤镜应用顺序：空间滤波 -> 时间滤波 -> 孔洞填充
    """
    
    # 空间滤波参数
    spatial_enabled: bool = True
    spatial_magnitude: int = 2        # 1-5, 值越大平滑越强
    spatial_smooth_alpha: float = 0.5 # 0-1, 平滑强度
    spatial_smooth_delta: int = 20    # 1-50, 边缘保持阈值
    spatial_holes_fill: int = 1       # 0-6, 孔洞填充级别
    
    # 时间滤波参数
    temporal_enabled: bool = True
    temporal_smooth_alpha: float = 0.4  # 0-1, 时间平滑强度
    temporal_smooth_delta: int = 20     # 1-100, 持久性阈值
    temporal_holes_fill: int = 1        # 0-6, 孔洞填充级别
    
    # 孔洞填充参数
    hole_filling_enabled: bool = True
    hole_filling_mode: int = 1  # 0=关闭, 1=最近邻(2像素), 2=远距离(4像素)


@dataclass
class SamplerConfig:
    """
    自适应深度采样器配置
    """
    
    # 默认采样参数
    default_window_size: int = 7
    default_foreground_ratio: float = 0.6
    default_method: str = 'median'  # median, mean, trimmed_mean, robust_mean
    
    # 时序一致性
    enable_temporal_check: bool = True
    max_change_ratio: float = 0.1  # 最大帧间变化率 (10%)
    history_size: int = 5
    
    # 离群值检测
    outlier_threshold: float = 0.03  # 3cm


@dataclass
class DepthProcessorConfig:
    """
    深度处理器配置
    """
    
    # 滤镜链配置
    filter_chain: FilterChainConfig = field(default_factory=FilterChainConfig)
    
    # 后处理
    enable_enhancement: bool = True
    bilateral_d: int = 5              # 双边滤波直径
    bilateral_sigma_color: float = 0.1  # 颜色空间标准差
    bilateral_sigma_space: float = 3.0  # 坐标空间标准差
    
    # 无效值填充
    enable_hole_filling: bool = True
    max_hole_size: int = 5  # 最大填充孔洞大小（像素）
    
    # 采样器配置
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    
    # 深度范围
    min_depth: float = 0.3  # 最小有效深度（米）
    max_depth: float = 4.0  # 最大有效深度（米）


# 身体部位采样策略
# 格式: (窗口大小, 前景比例, 统计方法, 离群阈值, 最大变化率)
BODY_PART_SAMPLING_STRATEGIES: Dict[str, tuple] = {
    # 头部（较小窗口，高前景比例）
    'nose': (5, 0.7, 'trimmed_mean', 0.02, 0.05),
    'left_eye': (3, 0.8, 'median', 0.01, 0.03),
    'right_eye': (3, 0.8, 'median', 0.01, 0.03),
    'left_ear': (5, 0.7, 'median', 0.02, 0.05),
    'right_ear': (5, 0.7, 'median', 0.02, 0.05),
    
    # 躯干（较大窗口，低前景比例）
    'left_shoulder': (11, 0.4, 'robust_mean', 0.03, 0.08),
    'right_shoulder': (11, 0.4, 'robust_mean', 0.03, 0.08),
    'left_hip': (13, 0.3, 'median', 0.04, 0.10),
    'right_hip': (13, 0.3, 'median', 0.04, 0.10),
    
    # 手臂（中等窗口）
    'left_elbow': (7, 0.6, 'median', 0.02, 0.06),
    'right_elbow': (7, 0.6, 'median', 0.02, 0.06),
    'left_wrist': (5, 0.7, 'trimmed_mean', 0.015, 0.04),
    'right_wrist': (5, 0.7, 'trimmed_mean', 0.015, 0.04),
    
    # 腿部
    'left_knee': (9, 0.5, 'robust_mean', 0.025, 0.07),
    'right_knee': (9, 0.5, 'robust_mean', 0.025, 0.07),
    'left_ankle': (7, 0.6, 'median', 0.02, 0.05),
    'right_ankle': (7, 0.6, 'median', 0.02, 0.05),
    
    # 脚部
    'left_heel': (5, 0.7, 'median', 0.015, 0.04),
    'right_heel': (5, 0.7, 'median', 0.015, 0.04),
    'left_foot_index': (3, 0.8, 'median', 0.01, 0.03),
    'right_foot_index': (3, 0.8, 'median', 0.01, 0.03),
}

# MediaPipe 关键点 ID 到身体部位名称的映射
KEYPOINT_ID_TO_BODY_PART: Dict[int, str] = {
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer',
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index',
}


def get_sampling_strategy(keypoint_id: int) -> tuple:
    """
    获取关键点的采样策略
    
    Args:
        keypoint_id: MediaPipe 关键点 ID (0-32)
    
    Returns:
        (窗口大小, 前景比例, 统计方法, 离群阈值, 最大变化率)
    """
    body_part = KEYPOINT_ID_TO_BODY_PART.get(keypoint_id)
    
    if body_part and body_part in BODY_PART_SAMPLING_STRATEGIES:
        return BODY_PART_SAMPLING_STRATEGIES[body_part]
    
    # 默认策略
    return (7, 0.6, 'median', 0.03, 0.06)


def load_depth_config(config_path: Optional[str] = None) -> DepthProcessorConfig:
    """
    从配置文件加载深度处理配置
    
    Args:
        config_path: 配置文件路径，默认为 config/depth_optimization.yaml
    
    Returns:
        DepthProcessorConfig 实例
    """
    if config_path is None:
        # 默认配置文件路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, 'config', 'depth_optimization.yaml')
    
    yaml_config = _load_yaml_config(config_path)
    
    if yaml_config is None:
        return DepthProcessorConfig()
    
    # 解析滤镜链配置
    filter_chain_config = FilterChainConfig()
    if 'filter_chain' in yaml_config:
        fc = yaml_config['filter_chain']
        
        # 空间滤波
        if 'spatial' in fc:
            spatial = fc['spatial']
            filter_chain_config.spatial_enabled = spatial.get('enabled', True)
            filter_chain_config.spatial_magnitude = spatial.get('magnitude', 2)
            filter_chain_config.spatial_smooth_alpha = spatial.get('smooth_alpha', 0.5)
            filter_chain_config.spatial_smooth_delta = spatial.get('smooth_delta', 20)
        
        # 时间滤波
        if 'temporal' in fc:
            temporal = fc['temporal']
            filter_chain_config.temporal_enabled = temporal.get('enabled', True)
            filter_chain_config.temporal_smooth_alpha = temporal.get('smooth_alpha', 0.4)
            filter_chain_config.temporal_smooth_delta = temporal.get('smooth_delta', 20)
        
        # 孔洞填充
        if 'hole_filling' in fc:
            hf = fc['hole_filling']
            filter_chain_config.hole_filling_enabled = hf.get('enabled', True)
            filter_chain_config.hole_filling_mode = hf.get('mode', 1)
    
    # 解析采样器配置
    sampler_config = SamplerConfig()
    if 'sampler' in yaml_config:
        sc = yaml_config['sampler']
        sampler_config.default_window_size = sc.get('default_window_size', 7)
        sampler_config.default_foreground_ratio = sc.get('default_foreground_ratio', 0.6)
        sampler_config.default_method = sc.get('default_method', 'median')
        
        # 时序一致性
        if 'temporal_consistency' in sc:
            tc = sc['temporal_consistency']
            sampler_config.enable_temporal_check = tc.get('enabled', True)
            sampler_config.max_change_ratio = tc.get('max_change_ratio', 0.1)
            sampler_config.history_size = tc.get('history_size', 5)
    
    # 解析深度处理器配置
    depth_config = DepthProcessorConfig(
        filter_chain=filter_chain_config,
        sampler=sampler_config
    )
    
    if 'depth_processor' in yaml_config:
        dp = yaml_config['depth_processor']
        depth_config.enable_enhancement = dp.get('enable_enhancement', True)
        depth_config.enable_hole_filling = dp.get('enable_hole_filling', True)
        depth_config.max_hole_size = dp.get('max_hole_size', 5)
        
        # 双边滤波
        if 'bilateral' in dp:
            bl = dp['bilateral']
            depth_config.bilateral_d = bl.get('d', 5)
            depth_config.bilateral_sigma_color = bl.get('sigma_color', 0.1)
            depth_config.bilateral_sigma_space = bl.get('sigma_space', 3.0)
    
    return depth_config
