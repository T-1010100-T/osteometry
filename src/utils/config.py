"""
配置管理模块
负责加载和管理系统配置
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# 全局配置实例
_config: Optional[Dict] = None


def get_project_root() -> Path:
    """获取项目根目录"""
    current = Path(__file__).resolve()
    # 向上查找直到找到 config 目录
    for parent in current.parents:
        if (parent / "config").exists():
            return parent
    return current.parent.parent.parent


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认使用 config/default.yaml
        """
        if config_path is None:
            config_path = get_project_root() / "config" / "default.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict = {}
        self._load()
    
    def _load(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项（支持点号分隔的路径）
        
        Args:
            key: 配置键，如 "camera.resolution.width"
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def config(self) -> Dict:
        """获取完整配置字典"""
        return self._config
    
    def __getitem__(self, key: str) -> Any:
        """支持字典风格访问"""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """支持 in 操作符"""
        return self.get(key) is not None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    获取全局配置实例（单例模式）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        ConfigLoader 实例
    """
    global _config
    
    if _config is None or config_path is not None:
        _config = ConfigLoader(config_path)
    
    return _config


def reload_config(config_path: Optional[str] = None) -> ConfigLoader:
    """重新加载配置"""
    global _config
    _config = None
    return get_config(config_path)


class HolisticConfigLoader(ConfigLoader):
    """
    Holistic 配置加载器
    
    专门用于加载和管理 holistic_settings.yaml 配置
    
    Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
    """
    
    # 默认配置值 (Requirements: 11.5)
    DEFAULTS = {
        'holistic': {
            'enabled': True,
            'model_complexity': 1,
            'smooth_landmarks': True,
            'enable_segmentation': False,
            'refine_face_landmarks': False
        },
        'hands': {
            'enabled': True,
            'detection_confidence': 0.7,
            'tracking_confidence': 0.5,
            'detection_interval': 1,
            'roi': {
                'enabled': True,
                'size': 300
            },
            'depth': {
                'median_filter_size': 5,
                'consistency_tolerance': 0.05
            },
            'filter': {
                'enabled': True,
                'window_size': 5,
                'use_kalman': True,
                'max_lost_frames': 5
            }
        },
        'gesture': {
            'enabled': True,
            'stability_threshold': 10,
            'confidence_threshold': 0.8,
            'thresholds': {
                'fist_curl_distance': 0.03,
                'open_extend_distance': 0.06,
                'ok_touch_distance': 0.02
            }
        },
        'measurement': {
            'min_confidence': 0.7,
            'filter_window': 5
        },
        'performance': {
            'adaptive_interval': True,
            'latency_threshold': 120,
            'fallback_interval': 5
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 Holistic 配置加载器
        
        Args:
            config_path: 配置文件路径，默认使用 config/holistic_settings.yaml
        """
        if config_path is None:
            config_path = get_project_root() / "config" / "holistic_settings.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict = {}
        self._callbacks: Dict[str, list] = {}  # 配置变更回调
        
        self._load_with_defaults()
    
    def _load_with_defaults(self) -> None:
        """加载配置文件，缺失项使用默认值"""
        # 从默认值开始
        self._config = self._deep_copy(self.DEFAULTS)
        
        # 如果配置文件存在，合并配置
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f) or {}
                self._deep_merge(self._config, file_config)
            except Exception as e:
                # Requirements: 11.5 - 配置文件缺失时使用默认值
                pass
    
    def _deep_copy(self, d: Dict) -> Dict:
        """深拷贝字典"""
        import copy
        return copy.deepcopy(d)
    
    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """深度合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def reload(self) -> None:
        """
        重新加载配置
        
        Requirements: 11.3 - 支持不重启系统更新配置
        """
        old_config = self._deep_copy(self._config)
        self._load_with_defaults()
        
        # 触发变更回调
        self._trigger_callbacks(old_config, self._config)
    
    def register_callback(self, key: str, callback) -> None:
        """
        注册配置变更回调
        
        Args:
            key: 配置键
            callback: 回调函数，接收 (old_value, new_value) 参数
        """
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)
    
    def _trigger_callbacks(self, old_config: Dict, new_config: Dict) -> None:
        """触发配置变更回调"""
        for key, callbacks in self._callbacks.items():
            old_value = self._get_nested(old_config, key)
            new_value = self._get_nested(new_config, key)
            
            if old_value != new_value:
                for callback in callbacks:
                    try:
                        callback(old_value, new_value)
                    except Exception:
                        pass
    
    def _get_nested(self, d: Dict, key: str) -> Any:
        """获取嵌套字典值"""
        keys = key.split('.')
        value = d
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value
    
    # 便捷属性访问
    @property
    def holistic_enabled(self) -> bool:
        return self.get('holistic.enabled', True)
    
    @property
    def model_complexity(self) -> int:
        return self.get('holistic.model_complexity', 1)
    
    @property
    def hands_enabled(self) -> bool:
        return self.get('hands.enabled', True)
    
    @property
    def detection_confidence(self) -> float:
        return self.get('hands.detection_confidence', 0.7)
    
    @property
    def tracking_confidence(self) -> float:
        return self.get('hands.tracking_confidence', 0.5)
    
    @property
    def hand_detection_interval(self) -> int:
        return self.get('hands.detection_interval', 1)
    
    @property
    def gesture_enabled(self) -> bool:
        return self.get('gesture.enabled', True)
    
    @property
    def gesture_stability_threshold(self) -> int:
        return self.get('gesture.stability_threshold', 10)


# 全局 Holistic 配置实例
_holistic_config: Optional[HolisticConfigLoader] = None


def get_holistic_config(config_path: Optional[str] = None) -> HolisticConfigLoader:
    """
    获取全局 Holistic 配置实例
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        HolisticConfigLoader 实例
    """
    global _holistic_config
    
    if _holistic_config is None or config_path is not None:
        _holistic_config = HolisticConfigLoader(config_path)
    
    return _holistic_config


def reload_holistic_config() -> HolisticConfigLoader:
    """重新加载 Holistic 配置"""
    global _holistic_config
    if _holistic_config:
        _holistic_config.reload()
    else:
        _holistic_config = HolisticConfigLoader()
    return _holistic_config
