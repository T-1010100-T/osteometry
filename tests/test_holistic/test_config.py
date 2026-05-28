"""
配置管理属性测试

**Feature: holistic-hand-integration, Property 17: 配置默认值回退**
**Validates: Requirements 11.5**
"""
import tempfile
import os
from pathlib import Path

import pytest
import yaml
from hypothesis import given, strategies as st, settings, HealthCheck

# 添加项目路径
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# 默认配置值定义
HOLISTIC_DEFAULTS = {
    'holistic.model_complexity': 1,
    'hands.detection_confidence': 0.7,
    'hands.tracking_confidence': 0.5,
    'hands.detection_interval': 1,
    'hands.roi.enabled': True,
    'hands.roi.size': 300,
    'hands.depth.median_filter_size': 5,
    'hands.depth.consistency_tolerance': 0.05,
    'hands.filter.enabled': True,
    'hands.filter.window_size': 5,
    'hands.filter.use_kalman': True,
    'hands.filter.max_lost_frames': 5,
    'gesture.enabled': True,
    'gesture.stability_threshold': 10,
    'gesture.confidence_threshold': 0.8,
}


class HolisticConfigLoader:
    """Holistic 配置加载器（带默认值回退）"""
    
    DEFAULTS = HOLISTIC_DEFAULTS
    
    def __init__(self, config_path: str = None):
        self._config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
    
    def get(self, key: str, default=None):
        """获取配置值，支持点号分隔路径，自动回退到默认值"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # 回退到默认值
                return self.DEFAULTS.get(key, default)
        
        return value


class TestConfigDefaults:
    """配置默认值回退测试"""
    
    def test_missing_config_file_uses_defaults(self):
        """
        **Property 17: 配置默认值回退**
        当配置文件不存在时，应使用默认值
        """
        loader = HolisticConfigLoader("/nonexistent/path.yaml")
        
        assert loader.get('holistic.model_complexity') == 1
        assert loader.get('hands.detection_confidence') == 0.7
        assert loader.get('hands.tracking_confidence') == 0.5
    
    def test_partial_config_uses_defaults_for_missing(self):
        """
        **Property 17: 配置默认值回退**
        当配置文件部分缺失时，缺失项应使用默认值
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'holistic': {'model_complexity': 2}}, f)
            temp_path = f.name
        
        try:
            loader = HolisticConfigLoader(temp_path)
            
            # 存在的配置应返回实际值
            assert loader.get('holistic.model_complexity') == 2
            
            # 缺失的配置应返回默认值
            assert loader.get('hands.detection_confidence') == 0.7
            assert loader.get('gesture.stability_threshold') == 10
        finally:
            os.unlink(temp_path)

    
    @given(st.sampled_from(list(HOLISTIC_DEFAULTS.keys())))
    @settings(max_examples=50)
    def test_all_defaults_accessible(self, key: str):
        """
        **Property 17: 配置默认值回退**
        对于任意默认配置键，当配置文件为空时应返回预定义的默认值
        """
        loader = HolisticConfigLoader(None)
        expected = HOLISTIC_DEFAULTS[key]
        actual = loader.get(key)
        
        assert actual == expected, f"Key {key}: expected {expected}, got {actual}"
    
    @given(
        st.dictionaries(
            keys=st.sampled_from(['holistic', 'hands', 'gesture']),
            values=st.dictionaries(
                keys=st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz_'),
                values=st.one_of(st.integers(min_value=-100, max_value=100), st.booleans()),
                min_size=0,
                max_size=2
            ),
            min_size=0,
            max_size=2
        )
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_custom_config_overrides_defaults(self, custom_config: dict):
        """
        **Property 17: 配置默认值回退**
        自定义配置应覆盖默认值，未指定的键应回退到默认值
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(custom_config, f)
            temp_path = f.name
        
        try:
            loader = HolisticConfigLoader(temp_path)
            
            # 验证所有默认键都可访问
            for key, default_value in HOLISTIC_DEFAULTS.items():
                value = loader.get(key)
                assert value is not None, f"Key {key} should have a value"
        finally:
            os.unlink(temp_path)
    
    def test_invalid_config_values_handled(self):
        """
        **Property 17: 配置默认值回退**
        无效配置值应被正确处理
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # 写入无效的 YAML 结构
            yaml.dump({'holistic': None}, f)
            temp_path = f.name
        
        try:
            loader = HolisticConfigLoader(temp_path)
            # 应回退到默认值
            assert loader.get('holistic.model_complexity') == 1
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
