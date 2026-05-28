# -*- coding: utf-8 -*-
"""
测量采集器模块

状态机管理和采样逻辑
"""
import os
import json
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from src.core.data_aggregator import (
    extract_required_fields,
    aggregate_samples,
    format_measurement_txt,
    to_cm_or_none,
    REQUIRED_MEASUREMENT_FIELDS_CN,
)


class CaptureState(Enum):
    """采集状态"""
    MONITORING = "monitoring"      # 监控中，等待站稳
    COUNTDOWN = "countdown"        # 倒计时中
    SAMPLING = "sampling"          # 采样中
    DONE = "done"                  # 完成


@dataclass
class CollectorConfig:
    """采集器配置"""
    required_stable_frames: int = 30
    countdown_seconds: float = 2.0
    sampling_window_seconds: float = 3.0
    sampling_target_frames: int = 10
    level_threshold: float = 3.0  # 水平阈值（度）
    output_dir: str = "data/sessions"


@dataclass
class CollectorState:
    """采集器状态"""
    state: CaptureState = CaptureState.MONITORING
    countdown_start_ts: Optional[float] = None
    sampling_start_ts: Optional[float] = None
    last_sample_ts: Optional[float] = None
    samples: List[Dict[str, float]] = field(default_factory=list)
    last_saved_path: Optional[str] = None
    saved_hint_until: float = 0.0
    frozen_display_frame: Optional[np.ndarray] = None
    
    # IMU 相关
    accel_data: Dict[str, float] = field(default_factory=lambda: {'x': 0.0, 'y': 0.0, 'z': 0.0})
    camera_pitch: float = 0.0
    camera_roll: float = 0.0
    is_camera_level: bool = True


class MeasurementCollector:
    """测量采集器"""
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.state = CollectorState()
        self._sampling_interval = self.config.sampling_window_seconds / float(self.config.sampling_target_frames)
    
    def reset(self):
        """重置状态"""
        self.state = CollectorState()
    
    def update_imu(self, imu_data: dict, alpha: float = 0.3):
        """更新 IMU 数据"""
        if 'accel' not in imu_data:
            return
        
        accel = imu_data['accel']
        # 低通滤波
        self.state.accel_data['x'] = alpha * accel['x'] + (1 - alpha) * self.state.accel_data['x']
        self.state.accel_data['y'] = alpha * accel['y'] + (1 - alpha) * self.state.accel_data['y']
        self.state.accel_data['z'] = alpha * accel['z'] + (1 - alpha) * self.state.accel_data['z']
        
        # 计算倾斜角度
        ax = self.state.accel_data['x']
        ay = self.state.accel_data['y']
        az = self.state.accel_data['z']
        
        if abs(ay) > 0.1 or abs(az) > 0.1:
            self.state.camera_pitch = math.degrees(math.atan2(ax, math.sqrt(ay*ay + az*az)))
            self.state.camera_roll = math.degrees(math.atan2(az, math.sqrt(ax*ax + ay*ay)))
            self.state.is_camera_level = (
                abs(self.state.camera_pitch) < self.config.level_threshold and 
                abs(self.state.camera_roll) < self.config.level_threshold
            )
    
    def update(self, timestamp: float, is_stable: bool, stable_progress: float,
               pose_detected: bool, measurement_values_cm: Optional[Dict],
               has_imu: bool = False, camera_mode: str = "opencv") -> Optional[Dict]:
        """
        更新状态机
        
        Returns:
            如果完成采集，返回保存结果信息，否则返回 None
        """
        # 相机是否准备好（水平）
        camera_ready = self.state.is_camera_level if (camera_mode == "realsense" and has_imu) else True
        
        if self.state.state == CaptureState.MONITORING:
            if (camera_ready and is_stable and stable_progress >= 1.0 and 
                pose_detected and measurement_values_cm is not None):
                self.state.state = CaptureState.COUNTDOWN
                self.state.countdown_start_ts = timestamp
                self.state.sampling_start_ts = None
                self.state.last_sample_ts = None
                self.state.samples = []
        
        elif self.state.state == CaptureState.COUNTDOWN:
            if not is_stable:
                self._reset_to_monitoring()
            elif self.state.countdown_start_ts is not None:
                if (timestamp - self.state.countdown_start_ts) >= self.config.countdown_seconds:
                    self.state.state = CaptureState.SAMPLING
                    self.state.sampling_start_ts = timestamp
                    self.state.last_sample_ts = None
                    self.state.samples = []
        
        elif self.state.state == CaptureState.SAMPLING:
            if not is_stable:
                self._reset_to_monitoring()
            else:
                # 按固定间隔采样
                can_sample = (
                    self.state.last_sample_ts is None or 
                    (timestamp - self.state.last_sample_ts) >= self._sampling_interval
                )
                if can_sample:
                    extracted = extract_required_fields(measurement_values_cm)
                    if extracted is not None:
                        self.state.samples.append(extracted)
                        self.state.last_sample_ts = timestamp
                
                sampling_elapsed = (
                    (timestamp - self.state.sampling_start_ts) 
                    if self.state.sampling_start_ts is not None else 0.0
                )
                
                # 检查是否完成采样
                if (len(self.state.samples) >= self.config.sampling_target_frames or 
                    sampling_elapsed >= self.config.sampling_window_seconds):
                    if self.state.samples:
                        return self._save_results(timestamp, camera_mode)
                    else:
                        self._reset_to_monitoring()
        
        return None
    
    def _reset_to_monitoring(self):
        """重置到监控状态"""
        self.state.state = CaptureState.MONITORING
        self.state.countdown_start_ts = None
        self.state.sampling_start_ts = None
        self.state.last_sample_ts = None
        self.state.samples = []
    
    def _save_results(self, timestamp: float, camera_mode: str) -> Dict:
        """保存采集结果"""
        aggregated, sample_counts = aggregate_samples(self.state.samples)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        
        img_path = os.path.join(self.config.output_dir, f"{timestamp_str}.jpg")
        txt_path = os.path.join(self.config.output_dir, f"{timestamp_str}.txt")
        json_path = os.path.join(self.config.output_dir, f"{timestamp_str}.json")
        
        meta = {
            "timestamp": datetime.now().isoformat(),
            "camera_mode": camera_mode,
            "stable_frames_required": self.config.required_stable_frames,
            "countdown_seconds": self.config.countdown_seconds,
            "sampling_window_seconds": self.config.sampling_window_seconds,
            "sampling_target_frames": self.config.sampling_target_frames,
            "sampling_frames_collected": len(self.state.samples),
            "method": "drop_min_max_then_mean",
        }
        
        payload: Dict = {"_meta": meta, "_sample_counts": sample_counts}
        payload.update(aggregated)
        
        txt_content = format_measurement_txt(meta=meta, aggregated_cm=aggregated)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        self.state.last_saved_path = json_path
        self.state.saved_hint_until = timestamp + 1.5
        self.state.state = CaptureState.DONE
        
        return {
            "img_path": img_path,
            "txt_path": txt_path,
            "json_path": json_path,
            "aggregated": aggregated,
            "meta": meta,
        }
    
    def manual_trigger(self, timestamp: float, is_stable: bool, stable_progress: float,
                       pose_detected: bool, measurement_values_cm: Optional[Dict],
                       has_imu: bool = False, camera_mode: str = "opencv"):
        """手动触发采集（按 'c' 键）"""
        camera_ready = self.state.is_camera_level if (camera_mode == "realsense" and has_imu) else True
        
        if (self.state.state in [CaptureState.MONITORING, CaptureState.DONE] and 
            camera_ready and is_stable and stable_progress >= 1.0 and 
            pose_detected and measurement_values_cm is not None):
            self.state.state = CaptureState.COUNTDOWN
            self.state.countdown_start_ts = timestamp
            self.state.sampling_start_ts = None
            self.state.last_sample_ts = None
            self.state.samples = []
    
    def get_state_message(self, timestamp: float, has_imu: bool, camera_mode: str,
                          stable_progress: float) -> str:
        """获取当前状态消息"""
        if self.state.state == CaptureState.MONITORING:
            if has_imu and not self.state.is_camera_level and camera_mode == "realsense":
                return "请先将相机放平"
            elif stable_progress >= 1.0:
                return "站稳检测通过，准备倒计时"
            else:
                return "请站稳以开始测量"
        
        elif self.state.state == CaptureState.COUNTDOWN:
            if self.state.countdown_start_ts is not None:
                remaining = max(0.0, self.config.countdown_seconds - (timestamp - self.state.countdown_start_ts))
                return f"倒计时: {remaining:.1f}s"
        
        elif self.state.state == CaptureState.SAMPLING:
            return f"采样中: {len(self.state.samples)}/{self.config.sampling_target_frames}"
        
        elif self.state.state == CaptureState.DONE:
            return "已保存"
        
        return ""
