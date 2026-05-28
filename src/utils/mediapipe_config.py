# -*- coding: utf-8 -*-
"""
MediaPipe 配置模块

解决中文路径问题：MediaPipe 的 C++ 层不支持中文路径
必须在导入 mediapipe 之前调用此模块

使用方法：
    from src.utils.mediapipe_config import ensure_mediapipe_env
    ensure_mediapipe_env()
    import mediapipe as mp
"""
import os
import sys
import tempfile

_ENV_CONFIGURED = [False]


def _get_cache_dir():
    cache_dir = os.environ.get('MEDIAPIPE_CACHE_DIR')
    if cache_dir and os.path.isdir(cache_dir):
        return cache_dir

    candidates = [
        os.path.join(tempfile.gettempdir(), 'mediapipe_cache'),
        'D:/mediapipe_cache',
        os.path.join(os.path.expanduser('~'), '.mediapipe_cache'),
    ]

    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            test_file = os.path.join(d, '.write_test')
            with open(test_file, 'w') as f:
                f.write('ok')
            os.remove(test_file)
            return d
        except (OSError, PermissionError):
            continue

    fallback = os.path.join(tempfile.gettempdir(), 'mediapipe_cache')
    os.makedirs(fallback, exist_ok=True)
    return fallback


def ensure_mediapipe_env():
    if _ENV_CONFIGURED[0]:
        return

    cache_dir = _get_cache_dir()

    os.environ['MEDIAPIPE_CACHE_DIR'] = cache_dir
    os.environ['XDG_CACHE_HOME'] = cache_dir
    os.environ['GLOG_log_dir'] = cache_dir
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    _ENV_CONFIGURED[0] = True


ensure_mediapipe_env()
