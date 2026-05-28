"""
相机控制器模块

管理 RealSense 和 OpenCV 相机的初始化、重启和帧获取
支持 RealSense 滤镜链优化深度数据
"""
import os
import time
from typing import Optional, Tuple, Any, List
from dataclasses import dataclass, field

import cv2
import numpy as np

# RealSense 可选导入
HAS_REALSENSE = True
try:
    import pyrealsense2 as rs
except ImportError:
    HAS_REALSENSE = False
    rs = None

from src.hardware.frame_set import Intrinsics
from src.core.coordinate_transformer import CoordinateTransformer
from src.core.hand_coordinate_transformer import HandCoordinateTransformer
from src.core.depth_config import FilterChainConfig, DepthProcessorConfig


def _env_float_or_none(name: str, default: Optional[float]) -> Optional[float]:
    """从环境变量获取浮点数，支持 None"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = str(raw).strip().lower()
    if raw in {"", "none", "null"}:
        return None
    try:
        return float(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    """从环境变量获取浮点数"""
    v = _env_float_or_none(name, default)
    try:
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)


@dataclass
class CameraConfig:
    """相机配置"""
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_imu: bool = False
    # 深度处理配置
    depth_config: Optional[DepthProcessorConfig] = None


@dataclass
class CameraState:
    """相机状态"""
    mode: str  # "realsense" | "opencv"
    pipeline: Any = None
    config: Any = None
    align: Any = None
    cap: Any = None
    transformer: Optional[CoordinateTransformer] = None
    hand_transformer: Optional[HandCoordinateTransformer] = None
    has_imu: bool = False
    timeout_count: int = 0
    # 滤镜链
    filters: List[Any] = field(default_factory=list)


class CameraController:
    """相机控制器"""
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.state: Optional[CameraState] = None
        # 确保有深度配置
        if self.config.depth_config is None:
            self.config.depth_config = DepthProcessorConfig()
    
    def start(self) -> CameraState:
        """启动相机，优先使用 RealSense"""
        if HAS_REALSENSE:
            try:
                state = self._start_realsense()
                print(f"✅ RealSense 相机启动成功")
                if state.has_imu:
                    print("✅ IMU 水平检测已启用")
                if state.filters:
                    print(f"✅ 深度滤镜链已启用 ({len(state.filters)} 个滤镜)")
                self.state = state
                return state
            except Exception as e:
                print(f"⚠️ RealSense 启动失败，回退到 OpenCV: {e}")
                self._cleanup_realsense()
                import time as _t
                _t.sleep(0.5)
        
        state = self._start_opencv()
        print(f"✅ OpenCV 相机启动成功")
        self.state = state
        return state

    def _cleanup_realsense(self):
        try:
            import gc
            gc.collect()
        except Exception:
            pass
    
    def _create_filter_chain(self) -> List[Any]:
        """创建 RealSense 滤镜链"""
        if not HAS_REALSENSE or rs is None:
            return []
        
        filters = []
        fc = self.config.depth_config.filter_chain
        
        # 空间滤波
        if fc.spatial_enabled:
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, fc.spatial_magnitude)
            spatial.set_option(rs.option.filter_smooth_alpha, fc.spatial_smooth_alpha)
            spatial.set_option(rs.option.filter_smooth_delta, fc.spatial_smooth_delta)
            spatial.set_option(rs.option.holes_fill, fc.spatial_holes_fill)
            filters.append(spatial)
        
        # 时间滤波
        if fc.temporal_enabled:
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, fc.temporal_smooth_alpha)
            temporal.set_option(rs.option.filter_smooth_delta, fc.temporal_smooth_delta)
            temporal.set_option(rs.option.holes_fill, fc.temporal_holes_fill)
            filters.append(temporal)
        
        # 孔洞填充
        if fc.hole_filling_enabled:
            hole_filling = rs.hole_filling_filter()
            hole_filling.set_option(rs.option.holes_fill, fc.hole_filling_mode)
            filters.append(hole_filling)
        
        return filters
    
    def _start_realsense(self) -> CameraState:
        """启动 RealSense 相机"""
        if not HAS_REALSENSE or rs is None:
            raise RuntimeError("RealSense not available")
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, self.config.fps)
        config.enable_stream(rs.stream.depth, self.config.width, self.config.height, rs.format.z16, self.config.fps)
        
        has_imu = False
        if self.config.enable_imu:
            try:
                config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
                has_imu = True
            except Exception:
                has_imu = False
        
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        
        # 创建滤镜链
        filters = self._create_filter_chain()
        
        # 获取内参
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        intrinsics = Intrinsics(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            ppx=intr.ppx,
            ppy=intr.ppy,
            coeffs=list(intr.coeffs)
        )
        
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        # 创建坐标变换器
        depth_sample_quantile = _env_float_or_none("MEAS_DEPTH_SAMPLE_QUANTILE", 0.5)
        depth_sample_trim_high = _env_float_or_none("MEAS_DEPTH_SAMPLE_TRIM_HIGH", 0.2)
        depth_consistency_tolerance = _env_float_or_none("MEAS_DEPTH_CONSIST_TOL", 0.15)
        hand_consistency_tolerance = _env_float_or_none("MEAS_HAND_CONSIST_TOL", 0.05)
        
        transformer = CoordinateTransformer(
            intrinsics,
            depth_scale=depth_scale,
            depth_consistency_tolerance=float(depth_consistency_tolerance) if depth_consistency_tolerance is not None else 0.15,
            depth_sample_quantile=depth_sample_quantile,
            depth_sample_trim_high=float(depth_sample_trim_high) if depth_sample_trim_high is not None else 0.0,
        )
        hand_transformer = HandCoordinateTransformer(
            intrinsics,
            depth_scale=depth_scale,
            consistency_tolerance=float(hand_consistency_tolerance) if hand_consistency_tolerance is not None else 0.05,
            depth_sample_quantile=depth_sample_quantile,
            depth_sample_trim_high=float(depth_sample_trim_high) if depth_sample_trim_high is not None else 0.0,
        )
        
        return CameraState(
            mode="realsense",
            pipeline=pipeline,
            config=config,
            align=align,
            transformer=transformer,
            hand_transformer=hand_transformer,
            has_imu=has_imu,
            filters=filters
        )
    
    def _start_opencv(self) -> CameraState:
        cap = None
        last_err = None

        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        if hasattr(cv2, 'CAP_MSMF'):
            backends.insert(0, cv2.CAP_MSMF)

        for backend in backends:
            backend_name = {cv2.CAP_DSHOW: 'DSHOW', cv2.CAP_ANY: 'ANY', cv2.CAP_MSMF: 'MSMF'}.get(backend, str(backend))
            for attempt in range(2):
                for idx in range(3):
                    try:
                        test_cap = cv2.VideoCapture(idx, backend)
                        if test_cap.isOpened():
                            test_cap.release()
                            cap = cv2.VideoCapture(idx, backend)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                            ret, _ = cap.read()
                            if ret:
                                print(f"✅ 摄像头索引 {idx} 启动成功 (backend={backend_name}, attempt={attempt + 1})")
                                break
                            else:
                                cap.release()
                                cap = None
                        else:
                            if test_cap is not None:
                                test_cap.release()
                    except Exception as e:
                        last_err = e
                        if cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = None
                        continue

                if cap is not None and cap.isOpened():
                    break

                if attempt < 1:
                    time.sleep(1.0)

            if cap is not None and cap.isOpened():
                break

            print(f"⚠️ 摄像头 backend={backend_name} 启动失败，尝试下一个后端...")

        if cap is None or not cap.isOpened():
            raise RuntimeError(f"OpenCV 摄像头启动失败: {last_err or '所有摄像头索引均不可用'}")

        intrinsics = Intrinsics(
            width=self.config.width,
            height=self.config.height,
            fx=self.config.width * 1.2,
            fy=self.config.height * 1.2,
            ppx=self.config.width / 2,
            ppy=self.config.height / 2
        )
        transformer = CoordinateTransformer(intrinsics=intrinsics, depth_scale=1.0)

        return CameraState(mode="opencv", cap=cap, transformer=transformer)
    
    def restart(self) -> Tuple[CameraState, Optional[Exception]]:
        """重启相机"""
        if self.state is None:
            return self.start(), None
        
        last_err = None
        for _ in range(3):
            try:
                self.stop()
            except Exception as e:
                last_err = e
            time.sleep(0.2)
            
            try:
                state = self.start()
                return state, None
            except Exception as e:
                last_err = e
                time.sleep(0.3)
        
        return self.state, last_err
    
    def stop(self):
        """停止相机"""
        if self.state is None:
            return
        
        if self.state.mode == "realsense" and self.state.pipeline:
            try:
                self.state.pipeline.stop()
            except Exception:
                pass
        elif self.state.mode == "opencv" and self.state.cap:
            try:
                self.state.cap.release()
            except Exception:
                pass
        
        self.state = None
    
    def get_frames(self, timeout_ms: int = 1000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """
        获取帧数据
        
        Returns:
            (color_image, depth_image, imu_data)
            imu_data: {'accel': {'x', 'y', 'z'}} 或空字典
        """
        if self.state is None:
            return None, None, {}
        
        if self.state.mode == "realsense":
            return self._get_realsense_frames(timeout_ms)
        else:
            return self._get_opencv_frames()
    
    def _get_realsense_frames(self, timeout_ms: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """获取 RealSense 帧"""
        try:
            frames = self.state.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError as e:
            if "Frame didn't arrive" in str(e):
                self.state.timeout_count += 1
                return None, None, {}
            raise
        
        if self.state.align is not None:
            frames = self.state.align.process(frames)
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            self.state.timeout_count += 1
            return None, None, {}
        
        self.state.timeout_count = 0
        
        # 应用滤镜链
        filtered_depth = depth_frame
        for f in self.state.filters:
            filtered_depth = f.process(filtered_depth)
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(filtered_depth.get_data())
        
        # 获取 IMU 数据
        imu_data = {}
        if self.state.has_imu:
            for frame in frames:
                if frame.is_motion_frame():
                    motion = frame.as_motion_frame()
                    if frame.get_profile().stream_type() == rs.stream.accel:
                        data = motion.get_motion_data()
                        imu_data['accel'] = {'x': data.x, 'y': data.y, 'z': data.z}
        
        return color_image, depth_image, imu_data
    
    def _get_opencv_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """获取 OpenCV 帧"""
        ok, color_image = self.state.cap.read()
        if not ok or color_image is None:
            return None, None, {}
        return color_image, None, {}
    
    @property
    def is_realsense(self) -> bool:
        """是否为 RealSense 模式"""
        return self.state is not None and self.state.mode == "realsense"
    
    @property
    def has_depth(self) -> bool:
        """是否有深度数据"""
        return self.is_realsense
    
    @property
    def has_imu(self) -> bool:
        """是否有 IMU"""
        return self.state is not None and self.state.has_imu
