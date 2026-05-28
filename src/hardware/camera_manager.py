"""
RealSense D455 相机管理器
负责相机初始化、配置和数据流管理
"""
from typing import Dict, Optional

import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

from .frame_set import FrameSet, Intrinsics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CameraError(Exception):
    """相机相关错误基类"""
    pass


class CameraNotFoundError(CameraError):
    """相机未找到"""
    pass


class CameraConnectionError(CameraError):
    """相机连接错误"""
    pass


class FrameAcquisitionError(CameraError):
    """帧获取错误"""
    pass


class CameraManager:
    """
    RealSense D455 相机管理器
    
    负责：
    - 相机初始化与配置
    - RGB-D 数据流管理
    - 深度滤波器配置
    - 相机内参获取
    
    Example:
        >>> camera = CameraManager()
        >>> camera.start()
        >>> frame = camera.get_frames()
        >>> camera.stop()
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化相机管理器
        
        Args:
            config: 相机配置字典，可包含:
                - resolution: {width, height}
                - fps: 帧率
                - depth: {preset, min_distance, max_distance}
                - filters: 滤波器配置
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 未安装，请运行: pip install pyrealsense2")
        
        self.config = config or self._default_config()
        
        self._pipeline: Optional[rs.pipeline] = None
        self._profile: Optional[rs.pipeline_profile] = None
        self._align: Optional[rs.align] = None
        self._filters: list = []
        self._intrinsics: Optional[Intrinsics] = None
        self._frame_count: int = 0
        self._running: bool = False
        
        logger.info("CameraManager 初始化完成")
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'resolution': {'width': 640, 'height': 480},
            'fps': 30,
            'depth': {
                'preset': 'high_accuracy',
                'min_distance': 0.3,
                'max_distance': 3.0
            },
            'filters': {
                'spatial': {'enabled': True, 'alpha': 0.5, 'delta': 20},
                'temporal': {'enabled': True, 'alpha': 0.4, 'delta': 20},
                'hole_filling': {'enabled': True, 'mode': 1}
            }
        }
    
    def start(self) -> bool:
        """
        启动相机数据流
        
        Returns:
            启动成功返回 True
        
        Raises:
            CameraNotFoundError: 未检测到相机
            CameraConnectionError: 连接失败
        """
        if self._running:
            logger.warning("相机已在运行")
            return True
        
        # 检查设备
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise CameraNotFoundError("未检测到 RealSense 设备")
        
        logger.info(f"检测到设备: {devices[0].get_info(rs.camera_info.name)}")
        
        try:
            # 创建管道
            self._pipeline = rs.pipeline()
            config = rs.config()
            
            # 配置流
            width = self.config['resolution']['width']
            height = self.config['resolution']['height']
            fps = self.config['fps']
            
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            
            # 启动管道
            self._profile = self._pipeline.start(config)
            
            # 配置深度传感器
            self._configure_depth_sensor()
            
            # 创建对齐器（深度对齐到彩色）
            self._align = rs.align(rs.stream.color)
            
            # 配置滤波器
            self._setup_filters()
            
            # 获取内参
            self._intrinsics = self._get_intrinsics()
            
            self._running = True
            self._frame_count = 0
            
            logger.info(f"相机启动成功: {width}x{height}@{fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"相机启动失败: {e}")
            raise CameraConnectionError(f"相机连接失败: {e}")
    
    def _configure_depth_sensor(self) -> None:
        """配置深度传感器"""
        depth_sensor = self._profile.get_device().first_depth_sensor()
        
        # 设置预设
        preset = self.config['depth'].get('preset', 'high_accuracy')
        preset_map = {
            'high_accuracy': rs.rs400_visual_preset.high_accuracy,
            'high_density': rs.rs400_visual_preset.high_density,
            'default': rs.rs400_visual_preset.default
        }
        if preset in preset_map:
            depth_sensor.set_option(rs.option.visual_preset, preset_map[preset])
            logger.debug(f"深度预设: {preset}")
    
    def _setup_filters(self) -> None:
        """配置深度滤波器"""
        self._filters = []
        filter_config = self.config.get('filters', {})
        
        # 空间滤波
        if filter_config.get('spatial', {}).get('enabled', True):
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_smooth_alpha, 
                             filter_config['spatial'].get('alpha', 0.5))
            spatial.set_option(rs.option.filter_smooth_delta, 
                             filter_config['spatial'].get('delta', 20))
            self._filters.append(spatial)
            logger.debug("启用空间滤波")
        
        # 时间滤波
        if filter_config.get('temporal', {}).get('enabled', True):
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha,
                              filter_config['temporal'].get('alpha', 0.4))
            temporal.set_option(rs.option.filter_smooth_delta,
                              filter_config['temporal'].get('delta', 20))
            self._filters.append(temporal)
            logger.debug("启用时间滤波")
        
        # 空洞填充
        if filter_config.get('hole_filling', {}).get('enabled', True):
            hole_filling = rs.hole_filling_filter()
            hole_filling.set_option(rs.option.holes_fill,
                                   filter_config['hole_filling'].get('mode', 1))
            self._filters.append(hole_filling)
            logger.debug("启用空洞填充")
    
    def _get_intrinsics(self) -> Intrinsics:
        """获取相机内参"""
        color_stream = self._profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        
        return Intrinsics(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            ppx=intr.ppx,
            ppy=intr.ppy,
            coeffs=list(intr.coeffs)
        )
    
    def stop(self) -> None:
        """停止相机数据流并释放资源"""
        if self._pipeline and self._running:
            self._pipeline.stop()
            self._running = False
            logger.info(f"相机停止，共处理 {self._frame_count} 帧")
    
    def get_frames(self, timeout_ms: int = 5000) -> FrameSet:
        """
        获取一帧 RGB-D 数据
        
        Args:
            timeout_ms: 超时时间（毫秒）
        
        Returns:
            FrameSet 对象
        
        Raises:
            FrameAcquisitionError: 帧获取失败
        """
        if not self._running:
            raise FrameAcquisitionError("相机未启动")
        
        try:
            # 等待帧
            frames = self._pipeline.wait_for_frames(timeout_ms)
            
            # 对齐
            aligned_frames = self._align.process(frames)
            
            # 获取帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                raise FrameAcquisitionError("获取帧数据为空")
            
            # 应用滤波器
            for f in self._filters:
                depth_frame = f.process(depth_frame)
            
            # 转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            self._frame_count += 1
            
            return FrameSet(
                color_frame=color_image,
                depth_frame=depth_image,
                timestamp=frames.get_timestamp() / 1000.0,  # 转换为秒
                frame_number=self._frame_count,
                intrinsics=self._intrinsics
            )
            
        except Exception as e:
            raise FrameAcquisitionError(f"帧获取失败: {e}")
    
    def get_intrinsics(self) -> Optional[Intrinsics]:
        """获取相机内参"""
        return self._intrinsics
    
    def is_connected(self) -> bool:
        """检查相机是否已连接"""
        ctx = rs.context()
        return len(ctx.query_devices()) > 0
    
    def is_running(self) -> bool:
        """检查相机是否正在运行"""
        return self._running
    
    @property
    def frame_count(self) -> int:
        """已处理的帧数"""
        return self._frame_count
    
    def __enter__(self):
        """支持 with 语句"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.stop()
        return False
