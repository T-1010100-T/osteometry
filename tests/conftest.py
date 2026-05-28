"""
pytest 配置和公共 fixtures
"""
import sys
from pathlib import Path

import pytest
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_color_frame():
    """生成测试用 RGB 图像"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_frame():
    """生成测试用深度图"""
    # 创建一个 1.5m 深度的平面
    return np.ones((480, 640), dtype=np.uint16) * 1500


@pytest.fixture
def sample_intrinsics():
    """测试用相机内参"""
    from src.hardware.frame_set import Intrinsics
    return Intrinsics(
        width=640,
        height=480,
        fx=600.0,
        fy=600.0,
        ppx=320.0,
        ppy=240.0,
        coeffs=[0.0] * 5
    )
