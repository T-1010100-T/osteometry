# -*- coding: utf-8 -*-
"""
身高识别系统 Web 应用

Flask 后端服务，提供：
- 静态文件服务
- WebSocket 实时视频流
- REST API 接口

运行方式：
    python app.py
"""
import os
import sys

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_APP_ROOT)

_lib_path = os.path.join(_APP_ROOT, 'lib')
if os.path.exists(_lib_path):
    sys.path.insert(0, _lib_path)

from src.utils.mediapipe_config import ensure_mediapipe_env
ensure_mediapipe_env()

import io
import base64
import json
import time
import threading
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from queue import Queue

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from flask import Flask, render_template, send_from_directory, send_file, jsonify, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.camera_controller import CameraController, CameraConfig
from src.core.measurement_engine import MeasurementEngine
from src.core.stability_detector import StabilityDetector
from src.core.measurement_collector import MeasurementCollector, CollectorConfig
from src.core.depth_config import load_depth_config
from src.core import HolisticEstimator
from src.core.constants import POSE_CONNECTIONS, HAND_CONNECTIONS
from src.core.keypoint_stabilizer import KeypointStabilizer
from src.core.stabilizer_config import StabilizerConfig

_WEB_DIR = os.path.join(_APP_ROOT, 'web')

app = Flask(__name__, 
            static_folder=_WEB_DIR,
            template_folder=_WEB_DIR)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

POSE_IGNORED_LANDMARKS = set(range(1, 11)) | set(range(17, 23))

@dataclass
class MeasurementRecord:
    time: str
    height: float
    shoulder: float
    arm_span: float
    leg_length: float
    sitting_height: float
    confidence: float
    pelvic_width: float = 0.0
    upper_limb_length: float = 0.0
    lower_limb_length: float = 0.0
    trunk_length: float = 0.0
    hand_length: float = 0.0
    foot_length: float = 0.0
    status: str = "success"
    image_path: Optional[str] = None

@dataclass
class AppState:
    camera: Optional[CameraController] = None
    estimator: Optional[HolisticEstimator] = None
    measurement_engine: Optional[MeasurementEngine] = None
    stability_detector: Optional[StabilityDetector] = None
    keypoint_stabilizer: Optional[KeypointStabilizer] = None
    is_running: bool = False
    is_measuring: bool = False
    current_frame: Optional[np.ndarray] = None
    latest_result: Optional[Dict] = None
    records: List[MeasurementRecord] = None
    images: List[Dict] = None
    measurement_history: List[Dict] = None
    _prev_body_detected: bool = False
    _body_lost_time: float = 0.0
    _height_warmup_count: int = 0
    # 自动/手动采集模式
    auto_collect: bool = True
    _stable_start_time: float = 0.0
    _auto_collected: bool = False
    _collect_countdown: float = 0.0

    def __post_init__(self):
        if self.records is None:
            self.records = []
        if self.images is None:
            self.images = []
        if self.measurement_history is None:
            self.measurement_history = []

state = AppState()
frame_queue = Queue(maxsize=10)

DATA_DIR = os.path.join(_APP_ROOT, "data", "sessions")
IMAGES_DIR = os.path.join(_APP_ROOT, "data", "images")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def draw_skeleton(image: np.ndarray, result, image_width: int, image_height: int) -> np.ndarray:
    """绘制骨骼"""
    display = image.copy()
    
    if result.pose.detected:
        landmarks = result.pose.landmarks
        
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                if start_idx in POSE_IGNORED_LANDMARKS or end_idx in POSE_IGNORED_LANDMARKS:
                    continue
                
                skip_forearm = False
                if start_idx == 13 and end_idx == 15 and result.left_hand and result.left_hand.detected:
                    skip_forearm = True
                if start_idx == 14 and end_idx == 16 and result.right_hand and result.right_hand.detected:
                    skip_forearm = True
                
                if skip_forearm:
                    continue
                
                lm1, lm2 = landmarks[start_idx], landmarks[end_idx]
                if lm1.visibility > 0.1 and lm2.visibility > 0.1:
                    x1, y1 = int(lm1.x * image_width), int(lm1.y * image_height)
                    x2, y2 = int(lm2.x * image_width), int(lm2.y * image_height)
                    cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        for i, lm in enumerate(landmarks):
            if lm.visibility > 0.1:
                if i in POSE_IGNORED_LANDMARKS:
                    continue
                x, y = int(lm.x * image_width), int(lm.y * image_height)
                cv2.circle(display, (x, y), 5, (0, 255, 255), -1)
    
    if result.left_hand and result.left_hand.detected:
        hand_landmarks = result.left_hand.landmarks
        
        if result.pose.detected and len(result.pose.landmarks) > 13:
            elbow = result.pose.landmarks[13]
            hand_wrist = hand_landmarks[0]
            if elbow.visibility > 0.1:
                ex, ey = int(elbow.x * image_width), int(elbow.y * image_height)
                wx, wy = int(hand_wrist.x * image_width), int(hand_wrist.y * image_height)
                cv2.line(display, (ex, ey), (wx, wy), (255, 255, 0), 3)
        
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                lm1, lm2 = hand_landmarks[start_idx], hand_landmarks[end_idx]
                x1, y1 = int(lm1.x * image_width), int(lm1.y * image_height)
                x2, y2 = int(lm2.x * image_width), int(lm2.y * image_height)
                cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for lm in hand_landmarks:
            cv2.circle(display, (int(lm.x * image_width), int(lm.y * image_height)), 3, (0, 255, 0), -1)
    
    if result.right_hand and result.right_hand.detected:
        hand_landmarks = result.right_hand.landmarks
        
        if result.pose.detected and len(result.pose.landmarks) > 14:
            elbow = result.pose.landmarks[14]
            hand_wrist = hand_landmarks[0]
            if elbow.visibility > 0.1:
                ex, ey = int(elbow.x * image_width), int(elbow.y * image_height)
                wx, wy = int(hand_wrist.x * image_width), int(hand_wrist.y * image_height)
                cv2.line(display, (ex, ey), (wx, wy), (255, 255, 0), 3)
        
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                lm1, lm2 = hand_landmarks[start_idx], hand_landmarks[end_idx]
                x1, y1 = int(lm1.x * image_width), int(lm1.y * image_height)
                x2, y2 = int(lm2.x * image_width), int(lm2.y * image_height)
                cv2.line(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for lm in hand_landmarks:
            cv2.circle(display, (int(lm.x * image_width), int(lm.y * image_height)), 3, (0, 0, 255), -1)
    
    if not result.pose.detected:
        cv2.putText(display, "Waiting for body detection...", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    return display

def draw_measurement_overlay(image: np.ndarray, measurements: Dict) -> np.ndarray:
    """绘制测量数据叠加层"""
    display = image.copy()
    h, w = display.shape[:2]
    
    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (280, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
    
    y_offset = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    
    height_val = measurements.get('身高') or 0
    shoulder_val = measurements.get('肩宽') or 0
    arm_span_val = measurements.get('臂展') or 0
    
    if height_val > 0:
        cv2.putText(display, f"Height: {height_val:.1f} cm", 
                    (20, y_offset), font, font_scale, (0, 255, 0), 2)
        y_offset += 30
    
    if shoulder_val > 0:
        cv2.putText(display, f"Shoulder: {shoulder_val:.1f} cm", 
                    (20, y_offset), font, font_scale, color, 2)
        y_offset += 30
    
    if arm_span_val > 0:
        cv2.putText(display, f"Arm Span: {arm_span_val:.1f} cm", 
                    (20, y_offset), font, font_scale, color, 2)
        y_offset += 30
    
    if height_val > 0:
        cv2.putText(display, f"Confidence: 85%", 
                    (20, y_offset), font, font_scale, color, 2)
    
    return display

def frame_to_base64(frame: np.ndarray) -> str:
    """将帧转换为 base64 字符串"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')

def reject_outliers_iqr(values: List[float], iqr_factor: float = 1.5) -> List[float]:
    """
    使用IQR方法剔除离群值
    
    Args:
        values: 数值列表
        iqr_factor: IQR倍数阈值
    
    Returns:
        剔除离群值后的列表
    """
    if len(values) < 4:
        return values
    
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    if iqr <= 0:
        return values
    
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    
    filtered = [v for v in values if lower <= v <= upper]
    return filtered if filtered else values


def reject_outliers_mad(values: List[float], threshold: float = 3.0) -> List[float]:
    """
    使用MAD（中位数绝对偏差）方法剔除离群值
    对小样本更鲁棒
    
    Args:
        values: 数值列表
        threshold: MAD倍数阈值
    
    Returns:
        剔除离群值后的列表
    """
    if len(values) < 4:
        return values
    
    arr = np.array(values)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    
    if mad <= 0:
        return values
    
    modified_z = 0.6745 * (arr - median) / mad
    filtered = [v for v, z in zip(values, modified_z) if abs(z) < threshold]
    return filtered if filtered else values


class SimpleKalmanFilter1D:
    """
    简化的1D卡尔曼滤波器
    
    用于测量值的时域平滑，比EWMA更优：
    - 自动调整增益（测量噪声大时更依赖预测，噪声小时更依赖观测）
    - 估计误差协方差自动收敛
    - 适合身高这种缓慢变化或不变的量
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0,
                 initial_estimate: float = 0.0, initial_error: float = 100.0):
        self.Q = process_noise
        self.R = measurement_noise
        self.x = initial_estimate
        self.P = initial_error
        self._initialized = False
    
    def update(self, measurement: float) -> float:
        if not self._initialized:
            self.x = measurement
            self.P = self.R
            self._initialized = True
            return self.x
        
        self.P += self.Q
        
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        
        return self.x
    
    def reset(self):
        self.x = 0.0
        self.P = 100.0
        self._initialized = False


_kalman_filters: Dict[str, SimpleKalmanFilter1D] = {}


def _get_kalman(key: str, measurement_noise: float = 1.0) -> SimpleKalmanFilter1D:
    if key not in _kalman_filters:
        _kalman_filters[key] = SimpleKalmanFilter1D(
            process_noise=0.5,
            measurement_noise=measurement_noise,
        )
    return _kalman_filters[key]


def reset_kalman_filters():
    _kalman_filters.clear()


def smooth_measurements(measurements: Dict, history: List[Dict], window_size: int = 15) -> Dict:
    """
    对测量结果进行简单平滑处理
    
    Args:
        measurements: 当前测量结果
        history: 历史测量结果列表
        window_size: 滑动窗口大小
    
    Returns:
        平滑后的测量结果
    """
    if not measurements:
        return measurements
    
    history.append(measurements)
    if len(history) > window_size:
        history.pop(0)
    
    if len(history) < 2:
        return measurements
    
    keys = ['身高', '肩宽', '臂展', '腿长', '坐高',
            '骨盆宽', '上肢长', '下肢长', '颈臀长', '手长', '足长',
            '脊柱底部到中部', '脊柱中部到肩部', '脊柱肩部到头部',
            '左肩到左肘', '左肘到左腕', '右肩到右肘', '右肘到右腕',
            '左髋到左膝', '左膝到左踝', '右髋到右膝', '右膝到右踝',
            '脊柱肩部到左肩', '脊柱肩部到右肩']
    
    smoothed = {}
    n = len(history)
    ewma_weights = np.exp(np.linspace(-2, 0, n))
    ewma_weights = ewma_weights / ewma_weights.sum()
    
    for key in keys:
        raw_values = []
        for h in history:
            val = h.get(key)
            if val is not None and val > 0:
                raw_values.append(val)
        
        if not raw_values:
            if key in measurements:
                smoothed[key] = measurements[key]
            continue
        
        # EWMA加权平均
        m = len(raw_values)
        sub_weights = np.exp(np.linspace(-1, 0, m))
        sub_weights = sub_weights / sub_weights.sum()
        smoothed[key] = float(sum(w * v for w, v in zip(sub_weights, raw_values)))
    
    for key, val in measurements.items():
        if key not in smoothed:
            smoothed[key] = val
    
    return smoothed

def camera_thread():
    """摄像头线程"""
    global state
    
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    log_timer = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 30
    
    logger.info("摄像头线程启动")
    
    detect_log_counter = 0
    
    while state.is_running:
        try:
            if not state.camera or not state.camera.state:
                time.sleep(0.1)
                continue
            
            if state.estimator is None:
                try:
                    logger.info("Estimator 为空，立即重建...")
                    state.estimator = HolisticEstimator(
                        model_complexity=1,
                        min_detection_confidence=0.2,
                        min_tracking_confidence=0.2,
                        enable_hands=False
                    )
                    logger.info("Estimator 重建成功")
                except Exception as e:
                    logger.error(f"Estimator 重建失败: {e}")
                time.sleep(0.1)
                continue
            
            color_frame, depth_frame, imu_data = state.camera.get_frames()
            
            if color_frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.warning(f"连续 {consecutive_failures} 帧获取失败，尝试重启相机...")
                    try:
                        camera_state, err = state.camera.restart()
                        if err:
                            logger.error(f"相机重启失败: {err}")
                        else:
                            logger.info("相机重启成功")
                    except Exception as restart_err:
                        logger.error(f"相机重启异常: {restart_err}")
                    consecutive_failures = 0
                time.sleep(0.033)
                continue
            
            consecutive_failures = 0
            rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            
            if state.estimator:
                try:
                    result = state.estimator.detect(rgb_frame)
                    
                    detect_log_counter += 1
                    if not result.body_detected and detect_log_counter % 10 == 0:
                        logger.warning(
                            f"检测丢失 #{detect_log_counter}: "
                            f"body={result.body_detected}, "
                            f"pose_lm={len(result.pose.landmarks) if result.pose.landmarks else 0}, "
                            f"fallback={state.estimator._fallback_mode}"
                        )
                    elif detect_log_counter % 60 == 0:
                        logger.info(
                            f"检测状态: body={result.body_detected}, "
                            f"landmarks={len(result.pose.landmarks)}, "
                            f"fallback={state.estimator._fallback_mode}"
                        )
                    
                    display_frame = draw_skeleton(
                        rgb_frame, result,
                        rgb_frame.shape[1], rgb_frame.shape[0]
                    )
                    
                    current_time = time.time()
                    # 检测到新人出现（连续丢失超过1秒才算真正离开）
                    if result.body_detected and not state._prev_body_detected:
                        # 如果距离上次检测丢失不到1秒，视为闪烁，不重置
                        if state._body_lost_time > 0 and (current_time - state._body_lost_time) < 1.0:
                            logger.info("检测闪烁恢复，跳过滤波器重置")
                        else:
                            reset_kalman_filters()
                            state.measurement_history.clear()
                            state._height_warmup_count = 0
                            if state.keypoint_stabilizer:
                                state.keypoint_stabilizer.reset()
                            if state.measurement_engine:
                                state.measurement_engine.enable_keypoint_stabilization = False
                            state._stable_start_time = 0
                            state._auto_collected = False
                            state._collect_countdown = 0
                            logger.info("检测到新人出现，所有滤波器已重置")
                    # 记录检测丢失时间
                    if not result.body_detected and state._prev_body_detected:
                        state._body_lost_time = current_time
                    state._prev_body_detected = result.body_detected

                    if result.body_detected and state.measurement_engine:
                        try:
                            if state.stability_detector:
                                state.stability_detector.add_frame(result)

                            measurements = None
                            if depth_frame is not None:
                                measurements = state.measurement_engine.calculate_measurements(
                                    result, depth_frame,
                                    rgb_frame.shape[1], rgb_frame.shape[0],
                                    timestamp=current_time
                                )
                            else:
                                # 获取相机内参（用于像素→厘米转换）
                                cam_intrinsics = None
                                if state.camera and state.camera.state and state.camera.state.transformer:
                                    cam_intrinsics = state.camera.state.transformer.intrinsics
                                measurements = state.measurement_engine.calculate_measurements_from_world_landmarks(
                                    result, timestamp=current_time,
                                    image_width=rgb_frame.shape[1], image_height=rgb_frame.shape[0],
                                    intrinsics=cam_intrinsics
                                )
                            if measurements:
                                height_raw = measurements.get('身高') or 0
                                if not (50 < height_raw < 250):
                                    # 身高越界，用上次有效值继续 emit，防止前端冻结
                                    if state.latest_result:
                                        try:
                                            _m = state.latest_result
                                            socketio.emit('measurement', {
                                                'height': _m.get('身高') or 0,
                                                'shoulder': _m.get('肩宽') or 0,
                                                'arm_span': _m.get('臂展') or 0,
                                                'leg_length': _m.get('腿长') or 0,
                                                'sitting_height': _m.get('坐高') or 0,
                                                'confidence': 0.0
                                            })
                                            state._last_emit_time = current_time
                                        except Exception:
                                            pass
                                if height_raw > 0 and 50 < height_raw < 250:
                                    # 预热期：直接用原始值，跳过所有滤波
                                    if state._height_warmup_count < 5:
                                        state._height_warmup_count += 1
                                        # 第5帧预热结束，重新启用关键点稳定
                                        if state._height_warmup_count >= 5 and state.measurement_engine:
                                            state.measurement_engine.enable_keypoint_stabilization = True
                                            logger.info("预热结束，关键点稳定已恢复")
                                        state.latest_result = measurements
                                        emit_data = {
                                            'height': height_raw,
                                            'shoulder': measurements.get('肩宽') or 0,
                                            'arm_span': measurements.get('臂展') or 0,
                                            'leg_length': measurements.get('腿长') or 0,
                                            'sitting_height': measurements.get('坐高') or 0,
                                            'pelvic_width': measurements.get('骨盆宽') or 0,
                                            'upper_limb_length': measurements.get('上肢长') or 0,
                                            'lower_limb_length': measurements.get('下肢长') or 0,
                                            'trunk_length': measurements.get('颈臀长') or 0,
                                            'hand_length': measurements.get('手长') or 0,
                                            'foot_length': measurements.get('足长') or 0,
                                            'confidence': 85.0
                                        }
                                    else:
                                        # 正常滤波
                                        kf = _get_kalman('身高', measurement_noise=2.0)
                                        height_filtered = kf.update(height_raw)
                                        measurements['身高'] = height_filtered

                                        smoothed = smooth_measurements(
                                            measurements,
                                            state.measurement_history,
                                            window_size=5
                                        )
                                        state.latest_result = smoothed
                                        emit_data = {
                                            'height': smoothed.get('身高') or 0,
                                            'shoulder': smoothed.get('肩宽') or 0,
                                            'arm_span': smoothed.get('臂展') or 0,
                                            'leg_length': smoothed.get('腿长') or 0,
                                            'sitting_height': smoothed.get('坐高') or 0,
                                            'pelvic_width': smoothed.get('骨盆宽') or 0,
                                            'upper_limb_length': smoothed.get('上肢长') or 0,
                                            'lower_limb_length': smoothed.get('下肢长') or 0,
                                            'trunk_length': smoothed.get('颈臀长') or 0,
                                            'hand_length': smoothed.get('手长') or 0,
                                            'foot_length': smoothed.get('足长') or 0,
                                            'confidence': 85.0
                                        }

                                    try:
                                        socketio.emit('measurement', emit_data)
                                        state._last_emit_time = current_time
                                    except Exception:
                                        pass

                                    # === 自动采集逻辑 ===
                                    if state.auto_collect and state.stability_detector and state._height_warmup_count >= 5:
                                        stability_result = state.stability_detector.get_stability()
                                        full_skeleton = state.stability_detector.is_full_skeleton
                                        is_stable = stability_result.is_stable and full_skeleton

                                        if is_stable:
                                            if state._stable_start_time == 0:
                                                state._stable_start_time = current_time
                                                state._auto_collected = False
                                            elapsed = current_time - state._stable_start_time
                                            remaining = max(0, 3.0 - elapsed)
                                            state._collect_countdown = remaining
                                            try:
                                                socketio.emit('stability_status', {
                                                    'stable': True,
                                                    'skeleton': True,
                                                    'countdown': round(remaining, 1),
                                                    'progress': round(min(elapsed / 3.0, 1.0), 2),
                                                    'hand_movement': round(stability_result.hand_movement * 100, 1),
                                                    'foot_movement': round(stability_result.body_movement * 100, 1)
                                                })
                                            except Exception:
                                                pass
                                            # 稳定满3秒，自动采集
                                            if elapsed >= 3.0 and not state._auto_collected:
                                                state._auto_collected = True
                                                ok, msg, rec = _do_collect(method='auto')
                                                logger.info(f"自动采集: {msg}")
                                                try:
                                                    socketio.emit('auto_collected', {
                                                        'success': ok,
                                                        'message': msg,
                                                        'data': rec
                                                    })
                                                except Exception:
                                                    pass
                                        else:
                                            # 不稳定或骨骼不完整，重置计时
                                            if state._stable_start_time != 0:
                                                state._stable_start_time = 0
                                                state._collect_countdown = 0
                                            reason = 'skeleton' if not full_skeleton else 'movement'
                                            try:
                                                socketio.emit('stability_status', {
                                                    'stable': False,
                                                    'skeleton': full_skeleton,
                                                    'countdown': 0,
                                                    'progress': 0,
                                                    'reason': reason,
                                                    'hand_movement': round(stability_result.hand_movement * 100, 1) if stability_result.hand_movement >= 0 else -1,
                                                    'foot_movement': round(stability_result.body_movement * 100, 1) if stability_result.body_movement >= 0 else -1
                                                })
                                            except Exception:
                                                pass

                                    height_val = state.latest_result.get('身高') or 0
                                    if current_time - log_timer > 3:
                                        logger.info(f"测量: 身高={height_val:.1f}cm (原始={height_raw:.1f}cm, warmup={state._height_warmup_count})")
                                        log_timer = current_time
                                else:
                                    # 身高越界，跳过不更新，保持上次有效值
                                    if height_raw > 0 and current_time - log_timer > 5:
                                        logger.warning(f"异常身高值: {height_raw:.1f}cm, 跳过")
                                        log_timer = current_time
                            else:
                                # measurements=None（计算失败），用上次有效值
                                if state.latest_result and current_time - getattr(state, '_last_emit_time', 0) > 0.1:
                                    state._last_emit_time = current_time
                                    try:
                                        _m = state.latest_result
                                        socketio.emit('measurement', {
                                            'height': _m.get('身高') or 0,
                                            'shoulder': _m.get('肩宽') or 0,
                                            'arm_span': _m.get('臂展') or 0,
                                            'leg_length': _m.get('腿长') or 0,
                                            'sitting_height': _m.get('坐高') or 0,
                                            'confidence': 0.0
                                        })
                                    except Exception:
                                        pass
                        except Exception as e:
                            logger.error(f"Measurement error: {e}")
                    else:
                        # body_detected=False 或无引擎：仍发送最后已知数据，防止前端数字冻结
                        if state.latest_result and current_time - getattr(state, '_last_emit_time', 0) > 0.1:
                            state._last_emit_time = current_time
                            try:
                                _m = state.latest_result
                                socketio.emit('measurement', {
                                    'height': _m.get('身高') or 0,
                                    'shoulder': _m.get('肩宽') or 0,
                                    'arm_span': _m.get('臂展') or 0,
                                    'leg_length': _m.get('腿长') or 0,
                                    'sitting_height': _m.get('坐高') or 0,
                                    'confidence': 0.0
                                })
                            except Exception:
                                pass
                    
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.error(f"Estimator error: {e}")
                    display_frame = color_frame
            else:
                display_frame = color_frame
            
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = current_time
            
            
            state.current_frame = display_frame
            state._current_fps = fps
            
            if not frame_queue.full():
                frame_queue.put(display_frame)
                
        except Exception as e:
            logger.error(f"Camera thread error: {e}")
            time.sleep(0.1)
    
    logger.info("Camera thread stopped")

def broadcast_frames():
    """广播帧到 WebSocket"""
    while state.is_running:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                fps_val = getattr(state, '_current_fps', 0)
                socketio.emit('frame', {'data': frame_base64, 'fps': fps_val})
            else:
                time.sleep(0.01)
                continue
            time.sleep(0.02)
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return send_from_directory(_WEB_DIR, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(_WEB_DIR, path)

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_running': state.is_running,
        'is_measuring': state.is_measuring,
        'camera_available': state.camera is not None,
        'records_count': len(state.records)
    })

@app.route('/api/start', methods=['POST'])
def start_camera():
    global state
    
    if state.is_running:
        return jsonify({'success': False, 'message': '摄像头已在运行'})

    errors = []
    
    try:
        depth_config = load_depth_config()
    except Exception as e:
        errors.append(f"深度配置加载失败: {e}")
        from src.core.depth_config import DepthProcessorConfig
        depth_config = DepthProcessorConfig()

    camera_config = CameraConfig(
        width=640,
        height=480,
        fps=30,
        enable_imu=False,
        depth_config=depth_config
    )

    if state.estimator is None:
        try:
            logger.info("正在初始化 HolisticEstimator...")
            state.estimator = HolisticEstimator(
                model_complexity=1,
                min_detection_confidence=0.2,
                min_tracking_confidence=0.2,
                enable_hands=False
            )
            logger.info("HolisticEstimator 初始化完成")
        except Exception as e:
            errors.append(f"姿态估计器初始化失败: {e}")
            logger.warning(f"姿态估计器初始化失败，继续启动: {e}")
            state.estimator = None
    else:
        logger.info("复用已有 HolisticEstimator")

    try:
        state.camera = CameraController(camera_config)
        logger.info("正在启动相机...")
        camera_state = state.camera.start()
        logger.info(f"相机模式: {camera_state.mode}")
    except Exception as e:
        errors.append(f"相机启动失败: {e}")
        state.camera = None
        return jsonify({
            'success': False,
            'message': f'启动失败: {e}',
            'detail': '; '.join(errors)
        })

    try:
        from src.core.stabilizer_config import get_stabilizer_preset
        stabilizer_config = get_stabilizer_preset('measurement')
        stabilizer_config.history_size = 10
        state.keypoint_stabilizer = KeypointStabilizer(config=stabilizer_config)
        logger.info("KeypointStabilizer 初始化完成 (measurement preset)")
    except Exception as e:
        errors.append(f"关键点稳定器初始化失败: {e}")
        logger.warning(f"关键点稳定器初始化失败: {e}")
        state.keypoint_stabilizer = None

    try:
        state.measurement_engine = MeasurementEngine(
            transformer=camera_state.transformer,
            keypoint_stabilizer=state.keypoint_stabilizer
        )
        logger.info("MeasurementEngine 初始化完成")
    except Exception as e:
        errors.append(f"测量引擎初始化失败: {e}")
        logger.warning(f"测量引擎初始化失败: {e}")
        state.measurement_engine = None

    try:
        state.stability_detector = StabilityDetector(
            window_size=60,         # 2秒@30fps
            movement_threshold=0.018,  # 手脚移动标准差 < 1.8cm 视为静止
            min_confidence=0.3
        )
    except Exception as e:
        logger.warning(f"稳定性检测器初始化失败: {e}")
        state.stability_detector = None

    state.measurement_history = []
    reset_kalman_filters()
    state.is_running = True

    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=broadcast_frames, daemon=True).start()

    extra_warnings = '; '.join(errors) if errors else ''
    return jsonify({
        'success': True,
        'message': f'摄像头启动成功 ({camera_state.mode})',
        'mode': camera_state.mode,
        'warnings': extra_warnings
    })

@app.route('/api/stop', methods=['POST'])
def stop_camera():
    global state
    
    state.is_running = False
    
    if state.camera:
        state.camera.stop()
        state.camera = None
    
    state.measurement_engine = None
    state.stability_detector = None
    state.current_frame = None
    state.latest_result = None
    
    return jsonify({'success': True, 'message': '摄像头已停止'})


def _do_collect(method='manual'):
    """执行一次采集，返回 (success, message, record_dict_or_none)"""
    global state

    measurements = state.latest_result
    if not measurements or not measurements.get('身高'):
        return False, '未检测到人体', None

    try:
        def get_val(key, default=0):
            val = measurements.get(key, default)
            return val if val is not None else default

        height = get_val('身高', 0)

        shoulder_width = get_val('肩宽', 0)
        if shoulder_width == 0:
            left_shoulder = get_val('脊柱肩部到左肩', 0)
            right_shoulder = get_val('脊柱肩部到右肩', 0)
            shoulder_width = left_shoulder + right_shoulder

        arm_span = get_val('臂展', 0)
        if arm_span == 0:
            left_arm = get_val('左肩到左肘', 0) + get_val('左肘到左腕', 0) + get_val('左腕到左手', 0)
            right_arm = get_val('右肩到右肘', 0) + get_val('右肘到右腕', 0) + get_val('右腕到右手', 0)
            arm_span = left_arm + right_arm + shoulder_width

        leg_length = get_val('腿长', 0)
        if leg_length == 0:
            left_leg = get_val('左髋到左膝', 0) + get_val('左膝到左踝', 0)
            right_leg = get_val('右髋到右膝', 0) + get_val('右膝到右踝', 0)
            leg_length = (left_leg + right_leg) / 2 if (left_leg > 0 or right_leg > 0) else 0

        sitting_height = get_val('坐高', 0)
        if sitting_height == 0:
            torso = get_val('脊柱底部到中部', 0) + get_val('脊柱中部到肩部', 0)
            head = get_val('脊柱肩部到头部', 0)
            sitting_height = torso + head if (torso > 0 or head > 0) else 0

        confidence = 85.0 if height > 0 else 0

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        timestamp_iso = datetime.now().isoformat()

        frame_to_save = None
        if state.current_frame is not None:
            frame_to_save = state.current_frame.copy()

        os.makedirs(DATA_DIR, exist_ok=True)

        txt_path = os.path.join(DATA_DIR, f"{timestamp_str}.txt")
        json_path = os.path.join(DATA_DIR, f"{timestamp_str}.json")
        img_path = os.path.join(DATA_DIR, f"{timestamp_str}.jpg")

        meta = {
            "timestamp": timestamp_iso,
            "camera_mode": "web",
            "method": method,
        }

        payload = {"_meta": meta, "_sample_counts": {k: 1 for k in measurements.keys()}}
        payload.update(measurements)

        txt_content = format_session_txt(meta=meta, measurements=measurements)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if frame_to_save is not None:
            success, buf = cv2.imencode('.jpg', frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                with open(img_path, 'wb') as f:
                    f.write(buf.tobytes())
            else:
                img_path = None
        else:
            img_path = None

        record = MeasurementRecord(
            time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            height=height,
            shoulder=shoulder_width,
            arm_span=arm_span,
            leg_length=leg_length,
            sitting_height=sitting_height,
            confidence=confidence,
            pelvic_width=get_val('骨盆宽', 0),
            upper_limb_length=get_val('上肢长', 0),
            lower_limb_length=get_val('下肢长', 0),
            trunk_length=get_val('颈臀长', 0),
            hand_length=get_val('手长', 0),
            foot_length=get_val('足长', 0),
            image_path=f"{timestamp_str}.jpg" if frame_to_save is not None else None
        )

        state.records.insert(0, record)
        save_records()

        if frame_to_save is not None:
            state.images.insert(0, {
                'filename': f"{timestamp_str}.jpg",
                'time': record.time,
                'height': record.height
            })

        return True, '测量完成', asdict(record)

    except Exception as e:
        logger.error(f"Measurement error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, f'测量失败: {str(e)}', None


@app.route('/api/measure', methods=['POST'])
def take_measurement():
    global state

    if not state.is_running:
        return jsonify({'success': False, 'message': '摄像头未启动'})

    success, msg, record = _do_collect(method='manual')
    if success:
        return jsonify({'success': True, 'message': msg, 'data': record})
    return jsonify({'success': False, 'message': msg})


def format_session_txt(meta: dict, measurements: dict) -> str:
    """格式化会话TXT文件"""
    lines = [
        "=" * 50,
        "        身体测量数据（单位：cm）",
        "=" * 50,
        "",
        f"测量时间\t{meta['timestamp']}",
        f"相机模式\t{meta['camera_mode']}",
        "",
        "-" * 50,
    ]
    
    for key, val in measurements.items():
        if val is not None and val > 0:
            lines.append(f"{key}\t{val:.1f}cm")
        else:
            lines.append(f"{key}\tN/A")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({
        'success': True,
        'data': [asdict(r) for r in state.records]
    })

@app.route('/api/images', methods=['GET'])
def get_images():
    return jsonify({
        'success': True,
        'data': state.images
    })

@app.route('/api/image/<filename>', methods=['GET'])
def get_image(filename):
    from urllib.parse import unquote
    decoded_filename = unquote(filename)
    
    for directory in [DATA_DIR, IMAGES_DIR]:
        img_path = os.path.join(directory, decoded_filename)
        if os.path.exists(img_path):
            try:
                return send_file(img_path, mimetype='image/jpeg')
            except Exception as e:
                logger.error(f"send_file error: {e}")
                continue
    
    logger.warning(f"图片不存在: {decoded_filename}")
    return jsonify({'success': False, 'message': '图片不存在'}), 404

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """获取所有会话文件列表"""
    sessions = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith('.json'):
                json_path = os.path.join(DATA_DIR, f)
                try:
                    with open(json_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        
                        def get_val(key, default=0):
                            val = data.get(key, default)
                            return val if val is not None else default
                        
                        height = get_val('身高', 0)
                        
                        # 优先使用直接存储的肩宽，否则从骨骼计算
                        shoulder_width = get_val('肩宽', 0)
                        if shoulder_width == 0:
                            left_shoulder = get_val('脊柱肩部到左肩', 0)
                            right_shoulder = get_val('脊柱肩部到右肩', 0)
                            shoulder_width = left_shoulder + right_shoulder
                        
                        # 优先使用直接存储的臂展，否则从骨骼计算
                        arm_span = get_val('臂展', 0)
                        if arm_span == 0:
                            left_arm = get_val('左肩到左肘', 0) + get_val('左肘到左腕', 0) + get_val('左腕到左手', 0)
                            right_arm = get_val('右肩到右肘', 0) + get_val('右肘到右腕', 0) + get_val('右腕到右手', 0)
                            arm_span = left_arm + right_arm + shoulder_width
                        
                        # 优先使用直接存储的腿长，否则从骨骼计算
                        leg_length = get_val('腿长', 0)
                        if leg_length == 0:
                            left_leg = get_val('左髋到左膝', 0) + get_val('左膝到左踝', 0)
                            right_leg = get_val('右髋到右膝', 0) + get_val('右膝到右踝', 0)
                            leg_length = (left_leg + right_leg) / 2 if (left_leg > 0 or right_leg > 0) else 0
                        
                        # 优先使用直接存储的坐高，否则从骨骼计算
                        sitting_height = get_val('坐高', 0)
                        if sitting_height == 0:
                            torso = get_val('脊柱底部到中部', 0) + get_val('脊柱中部到肩部', 0)
                            head = get_val('脊柱肩部到头部', 0)
                            sitting_height = torso + head if (torso > 0 or head > 0) else 0
                        
                        sessions.append({
                            'filename': f,
                            'timestamp': data.get('_meta', {}).get('timestamp', ''),
                            'height': height,
                            'shoulder': shoulder_width,
                            'arm_span': arm_span,
                            'leg_length': leg_length,
                            'sitting_height': sitting_height,
                            'pelvic_width': get_val('骨盆宽', 0),
                            'upper_limb_length': get_val('上肢长', 0),
                            'lower_limb_length': get_val('下肢长', 0),
                            'trunk_length': get_val('颈臀长', 0),
                            'hand_length': get_val('手长', 0),
                            'foot_length': get_val('足长', 0),
                            'confidence': 85.0 if height > 0 else 0,
                            'has_image': os.path.exists(os.path.join(DATA_DIR, f.replace('.json', '.jpg')))
                        })
                except:
                    pass
    sessions.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({
        'success': True,
        'data': sessions
    })

@app.route('/api/session/<filename>', methods=['GET'])
def get_session(filename):
    """获取单个会话详情"""
    json_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(json_path):
        return jsonify({'success': False, 'message': '会话不存在'}), 404
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/session/<filename>', methods=['DELETE'])
def delete_session(filename):
    """删除单个会话"""
    try:
        base_name = filename.replace('.json', '')
        deleted_files = []
        
        json_path = os.path.join(DATA_DIR, f"{base_name}.json")
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted_files.append(f"{base_name}.json")
        
        txt_path = os.path.join(DATA_DIR, f"{base_name}.txt")
        if os.path.exists(txt_path):
            os.remove(txt_path)
            deleted_files.append(f"{base_name}.txt")
        
        img_path = os.path.join(DATA_DIR, f"{base_name}.jpg")
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted_files.append(f"{base_name}.jpg")
        
        logger.info(f"删除会话: {deleted_files}")
        return jsonify({
            'success': True,
            'message': f'已删除 {len(deleted_files)} 个文件',
            'deleted': deleted_files
        })
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/sessions/batch-delete', methods=['POST'])
def batch_delete_sessions():
    """批量删除会话"""
    try:
        data = request.get_json()
        filenames = data.get('filenames', [])
        
        if not filenames:
            return jsonify({'success': False, 'message': '未选择要删除的文件'})
        
        deleted_count = 0
        failed = []
        
        for filename in filenames:
            try:
                base_name = filename.replace('.json', '')
                
                json_path = os.path.join(DATA_DIR, f"{base_name}.json")
                if os.path.exists(json_path):
                    os.remove(json_path)
                
                txt_path = os.path.join(DATA_DIR, f"{base_name}.txt")
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                
                img_path = os.path.join(DATA_DIR, f"{base_name}.jpg")
                if os.path.exists(img_path):
                    os.remove(img_path)
                
                deleted_count += 1
            except Exception as e:
                failed.append({'filename': filename, 'error': str(e)})
        
        logger.info(f"批量删除: 成功 {deleted_count}, 失败 {len(failed)}")
        return jsonify({
            'success': True,
            'message': f'已删除 {deleted_count} 个会话',
            'deleted_count': deleted_count,
            'failed': failed
        })
    except Exception as e:
        logger.error(f"批量删除失败: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/image/<filename>', methods=['DELETE'])
def delete_image(filename):
    """删除单张图片及其关联数据"""
    from urllib.parse import unquote
    try:
        decoded_filename = unquote(filename)
        base_name = decoded_filename.replace('.jpg', '').replace('.json', '')
        
        deleted_files = []
        
        for ext in ['.jpg', '.json', '.txt']:
            file_path = os.path.join(DATA_DIR, f"{base_name}{ext}")
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(f"{base_name}{ext}")
        
        if deleted_files:
            logger.info(f"删除图片: {deleted_files}")
            return jsonify({
                'success': True,
                'message': f'已删除 {len(deleted_files)} 个文件',
                'deleted_files': deleted_files
            })
        else:
            return jsonify({'success': False, 'message': '文件不存在'}), 404
            
    except Exception as e:
        logger.error(f"删除图片失败: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/sessions/delete-all', methods=['POST'])
def delete_all_sessions():
    """删除所有会话"""
    try:
        deleted_count = 0
        
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                if f.endswith(('.json', '.txt', '.jpg')):
                    try:
                        os.remove(os.path.join(DATA_DIR, f))
                        deleted_count += 1
                    except:
                        pass
        
        logger.info(f"删除所有会话: {deleted_count} 个文件")
        return jsonify({
            'success': True,
            'message': f'已删除 {deleted_count} 个文件',
            'deleted_count': deleted_count
        })
    except Exception as e:
        logger.error(f"删除所有会话失败: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    global state
    
    data = request.get_json()
    actual_height = data.get('actual_height')
    
    if not actual_height:
        return jsonify({'success': False, 'message': '请提供实际身高'})
    
    if not state.latest_result or not state.latest_result.get('身高'):
        return jsonify({'success': False, 'message': '请先进行测量'})
    
    measured_height = state.latest_result['身高']
    scale = actual_height / measured_height
    
    os.environ['MEAS_HEIGHT_SCALE'] = str(scale)
    os.environ['MEAS_LINEAR_SCALE'] = str(scale)
    
    return jsonify({
        'success': True,
        'message': f'校准完成，系数: {scale:.4f}',
        'scale': scale
    })

@app.route('/api/delete/<int:index>', methods=['DELETE'])
def delete_record(index):
    global state
    
    if 0 <= index < len(state.records):
        record = state.records.pop(index)
        if record.image_path:
            try:
                img_path = os.path.join(DATA_DIR, record.image_path)
                if not os.path.exists(img_path):
                    img_path = os.path.join(IMAGES_DIR, record.image_path)
                if os.path.exists(img_path):
                    os.remove(img_path)
                state.images = [img for img in state.images if img['filename'] != record.image_path]
            except:
                pass
        save_records()
        return jsonify({'success': True, 'message': '记录已删除'})
    
    return jsonify({'success': False, 'message': '记录不存在'})

@app.route('/api/export', methods=['GET'])
def export_data():
    if not state.records:
        return jsonify({'success': False, 'message': '没有可导出的数据'})
    
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['时间', '身高(cm)', '肩宽(cm)', '臂展(cm)', '腿长(cm)', '坐高(cm)', '置信度(%)'])
    
    for record in state.records:
        writer.writerow([
            record.time, record.height, record.shoulder,
            record.arm_span, record.leg_length, record.sitting_height,
            record.confidence
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=measurements.csv'}
    )

def save_records():
    try:
        filepath = os.path.join(DATA_DIR, 'records.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in state.records], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Save records error: {e}")

def load_records():
    global state
    try:
        filepath = os.path.join(DATA_DIR, 'records.json')
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                state.records = [MeasurementRecord(**r) for r in data]
                
                for record in state.records:
                    if record.image_path:
                        img_file = os.path.join(DATA_DIR, record.image_path)
                        if not os.path.exists(img_file):
                            img_file = os.path.join(IMAGES_DIR, record.image_path)
                        if os.path.exists(img_file) and os.path.getsize(img_file) > 0:
                            state.images.append({
                                'filename': record.image_path,
                                'time': record.time,
                                'height': record.height
                            })
                        else:
                            record.image_path = None
    except Exception as e:
        logger.error(f"Load records error: {e}")

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'is_running': state.is_running})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_stream')
def handle_start_stream(data=None):
    if not state.is_running:
        emit('error', {'message': '摄像头未启动'})

@socketio.on('set_collect_mode')
def handle_set_collect_mode(data):
    """切换自动/手动采集模式"""
    mode = data.get('mode', 'auto')
    state.auto_collect = (mode == 'auto')
    state._stable_start_time = 0
    state._auto_collected = False
    state._collect_countdown = 0
    logger.info(f"采集模式切换: {'自动' if state.auto_collect else '手动'}")
    emit('collect_mode_changed', {'mode': mode})

@socketio.on('manual_collect')
def handle_manual_collect():
    """手动触发采集（SocketIO 通道）"""
    ok, msg, rec = _do_collect(method='manual')
    emit('collect_result', {'success': ok, 'message': msg, 'data': rec})

def print_banner():
    print("\n" + "=" * 60)
    print("   身高识别系统 - Web 服务")
    print("=" * 60)
    print()
    print("   访问地址: http://localhost:5000")
    print()
    print("   功能:")
    print("     • 实时摄像头预览")
    print("     • 身高自动识别")
    print("     • 历史数据查看")
    print("     • 测量图片管理")
    print()
    print("   按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")

def preinit_estimator():
    """预初始化姿态估计器，确保启动即可用"""
    if state.estimator is None:
        try:
            logger.info("预初始化 HolisticEstimator...")
            state.estimator = HolisticEstimator(
                model_complexity=1,
                min_detection_confidence=0.2,
                min_tracking_confidence=0.2,
                enable_hands=False
            )
            logger.info("预初始化 HolisticEstimator 完成")
        except Exception as e:
            logger.warning(f"预初始化 HolisticEstimator 失败，将在启动摄像头时重试: {e}")

def main():
    preinit_estimator()
    load_records()
    print_banner()
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        if state.is_running:
            state.is_running = False
            if state.camera:
                state.camera.stop()

if __name__ == '__main__':
    main()
