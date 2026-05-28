"""
日志模块
基于 loguru 的统一日志配置
"""
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


# 移除默认 handler
logger.remove()

# 是否已初始化
_initialized = False


def setup_logger(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    """
    配置日志系统
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_dir: 日志文件目录
        console: 是否输出到控制台
        file: 是否输出到文件
        rotation: 日志轮转大小
        retention: 日志保留时间
    """
    global _initialized
    
    if _initialized:
        return
    
    # 日志格式
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    
    # 控制台输出
    if console:
        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=True
        )
    
    # 文件输出
    if file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 常规日志
        logger.add(
            log_path / "app_{time:YYYY-MM-DD}.log",
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )
        
        # 错误日志单独记录
        logger.add(
            log_path / "error_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="ERROR",
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )
    
    _initialized = True
    logger.info(f"日志系统初始化完成 (level={level})")


def get_logger(name: str = None):
    """
    获取 logger 实例
    
    Args:
        name: 模块名称
    
    Returns:
        logger 实例
    """
    global _initialized
    
    if not _initialized:
        # 使用默认配置初始化
        setup_logger()
    
    if name:
        return logger.bind(name=name)
    return logger
