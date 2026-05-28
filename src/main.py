# -*- coding: utf-8 -*-
"""
CycleFitting 主程序入口

身体测量系统 - 使用 MediaPipe 进行实时姿态估计

按键:
  q - 退出
  r - 重启摄像头
  c - 手动触发采集
"""
import sys
import os
import time
from typing import Optional

# 设置编码
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.utils.mediapipe_config import ensure_mediapipe_env
ensure_mediapipe_env()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

from src.core.camera_controller import CameraController, CameraConfig
from src.core.measurement_collector import MeasurementCollector, CollectorConfig, CaptureState
from src.core.measurement_engine import MeasurementEngine
from src.core.stability_detector import StabilityDetector
from src.core.constants import POSE_CONNECTIONS, HAND_CONNECTIONS
from src.core.depth_processor import DepthProcessor
from src.core.depth_config import DepthProcessorConfig, load_depth_config
from src.core.keypoint_stabilizer import KeypointStabilizer
from src.core.stabilizer_config import StabilizerConfig
from src.core.biomechanical_constraints import BiomechanicalConstraints, ConstraintsConfig
from src.visualization.ui_renderer import (
    draw_chinese_text,
    draw_progress_bar,
    draw_level_indicator,
    render_measurement_panel
)


# 忽略的姿态关键点（面部细节）
POSE_IGNORED_LANDMARKS = set(range(1, 11)) | set(range(17, 23))


def draw_skeleton(image: np.ndarray, result, image_width: int, image_height: int):
    """绘制骨骼"""
    # 身体
    if result.pose.detected:
        landmarks = result.pose.landmarks
        
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                if start_idx in POSE_IGNORED_LANDMARKS or end_idx in POSE_IGNORED_LANDMARKS:
                    continue
                
                # 如果是小臂连接（肘到腕），且有手部检测，跳过身体的手腕点
                # 左小臂: 13->15, 右小臂: 14->16
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
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        for i, lm in enumerate(landmarks):
            if lm.visibility > 0.1:
                if i in POSE_IGNORED_LANDMARKS:
                    continue
                x, y = int(lm.x * image_width), int(lm.y * image_height)
                cv2.circle(image, (x, y), 5, (0, 255, 255), -1)
    
    # 左手 - 绘制肘部到手部手腕的连接
    if result.left_hand and result.left_hand.detected:
        hand_landmarks = result.left_hand.landmarks
        
        # 绘制肘部到手部手腕的小臂骨骼（青色，更粗）
        if result.pose.detected and len(result.pose.landmarks) > 13:
            elbow = result.pose.landmarks[13]  # 左肘
            hand_wrist = hand_landmarks[0]  # 手部手腕
            if elbow.visibility > 0.1:
                ex, ey = int(elbow.x * image_width), int(elbow.y * image_height)
                wx, wy = int(hand_wrist.x * image_width), int(hand_wrist.y * image_height)
                cv2.line(image, (ex, ey), (wx, wy), (255, 255, 0), 3)
        
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                lm1, lm2 = hand_landmarks[start_idx], hand_landmarks[end_idx]
                x1, y1 = int(lm1.x * image_width), int(lm1.y * image_height)
                x2, y2 = int(lm2.x * image_width), int(lm2.y * image_height)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for lm in hand_landmarks:
            cv2.circle(image, (int(lm.x * image_width), int(lm.y * image_height)), 3, (0, 255, 0), -1)
    
    if result.right_hand and result.right_hand.detected:
        hand_landmarks = result.right_hand.landmarks
        
        if result.pose.detected and len(result.pose.landmarks) > 14:
            elbow = result.pose.landmarks[14]
            hand_wrist = hand_landmarks[0]
            if elbow.visibility > 0.1:
                ex, ey = int(elbow.x * image_width), int(elbow.y * image_height)
                wx, wy = int(hand_wrist.x * image_width), int(hand_wrist.y * image_height)
                cv2.line(image, (ex, ey), (wx, wy), (255, 255, 0), 3)
        
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                lm1, lm2 = hand_landmarks[start_idx], hand_landmarks[end_idx]
                x1, y1 = int(lm1.x * image_width), int(lm1.y * image_height)
                x2, y2 = int(lm2.x * image_width), int(lm2.y * image_height)
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for lm in hand_landmarks:
            cv2.circle(image, (int(lm.x * image_width), int(lm.y * image_height)), 3, (0, 0, 255), -1)


def main():
    """主函数"""
    print("=" * 50)
    print("CycleFitting 身体测量系统")
    print("=" * 50)
    print()
    print("按键:")
    print("  q - 退出")
    print("  r - 重启摄像头")
    print("  c - 手动触发采集")
    print("  h - 设置身高校准")
    print()
    
    # 导入 MediaPipe 估计器
    from src.core import HolisticEstimator
    print("✅ MediaPipe 可用")
    
    # 初始化相机（使用深度优化配置）
    depth_config = load_depth_config()
    camera_config = CameraConfig(
        width=640, 
        height=480, 
        fps=30, 
        enable_imu=False,
        depth_config=depth_config
    )
    camera = CameraController(camera_config)
    
    try:
        camera_state = camera.start()
    except Exception as e:
        print(f"❌ 相机启动失败: {e}")
        input("按回车键退出...")
        return 1
    
    print(f"✅ 相机启动成功! ({camera_state.mode})")
    
    # 初始化组件
    estimator = HolisticEstimator()
    stability_detector = StabilityDetector(window_size=10)
    
    collector_config = CollectorConfig(
        required_stable_frames=30,
        countdown_seconds=2.0,
        sampling_window_seconds=3.0,
        sampling_target_frames=10,
        level_threshold=3.0,
        output_dir="data/sessions"
    )
    collector = MeasurementCollector(collector_config)
    
    # 测量引擎
    measurement_engine: Optional[MeasurementEngine] = None
    depth_processor: Optional[DepthProcessor] = None
    keypoint_stabilizer: Optional[KeypointStabilizer] = None
    biomechanical_constraints: Optional[BiomechanicalConstraints] = None

    if camera_state.transformer:
        if camera.is_realsense:
            depth_processor_config = DepthProcessorConfig(
                enable_enhancement=False,
                enable_hole_filling=False
            )
            depth_processor = DepthProcessor(depth_processor_config)

            stabilizer_config = StabilizerConfig()
            keypoint_stabilizer = KeypointStabilizer(stabilizer_config)

            constraints_config = ConstraintsConfig()
            biomechanical_constraints = BiomechanicalConstraints(constraints_config)

            measurement_engine = MeasurementEngine(
                camera_state.transformer,
                camera_state.hand_transformer,
                depth_processor,
                keypoint_stabilizer,
                biomechanical_constraints
            )
            print("✅ RealSense 滤镜链已启用")
            print("✅ 关键点稳定已启用")
            print("✅ 生物力学约束已启用")
        else:
            # OpenCV 模式：用 world_landmarks 计算，仅需基础引擎
            measurement_engine = MeasurementEngine(
                camera_state.transformer,
                depth_processor=None,
                keypoint_stabilizer=None,
                biomechanical_constraints=None
            )
            print("✅ OpenCV 模式：使用 world_landmarks 测量")
    
    # 状态变量
    frame_count = 0
    fps_start = time.time()
    fps = 0.0
    last_measurement_ts = 0.0
    measurement_values_cm = None
    panel_w = 420
    
    window_name = 'CycleFitting - 身体测量'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, 640 + panel_w, 480)
    
    try:
        while True:
            # 完成状态：显示冻结帧
            if collector.state.state == CaptureState.DONE and collector.state.frozen_display_frame is not None:
                cv2.imshow(window_name, collector.state.frozen_display_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    camera_state, err = camera.restart()
                    if err:
                        print(f"⚠️ 相机重启失败: {err}")
                    stability_detector.reset()
                    collector.reset()
                    measurement_values_cm = None
                    if camera_state.transformer:
                        if camera.is_realsense:
                            if depth_processor:
                                depth_processor.reset()
                            else:
                                depth_processor_config = DepthProcessorConfig(
                                    enable_enhancement=False,
                                    enable_hole_filling=False
                                )
                                depth_processor = DepthProcessor(depth_processor_config)

                            if keypoint_stabilizer:
                                keypoint_stabilizer.reset()
                            else:
                                stabilizer_config = StabilizerConfig()
                                keypoint_stabilizer = KeypointStabilizer(stabilizer_config)

                            if not biomechanical_constraints:
                                constraints_config = ConstraintsConfig()
                                biomechanical_constraints = BiomechanicalConstraints(constraints_config)

                            measurement_engine = MeasurementEngine(
                                camera_state.transformer,
                                camera_state.hand_transformer,
                                depth_processor,
                                keypoint_stabilizer,
                                biomechanical_constraints
                            )
                        else:
                            measurement_engine = MeasurementEngine(
                                camera_state.transformer,
                                depth_processor=None,
                                keypoint_stabilizer=None,
                                biomechanical_constraints=None
                            )
                continue
            
            # 获取帧
            color_image, depth_image, imu_data = camera.get_frames()
            if color_image is None:
                if camera_state.timeout_count >= 5:
                    camera_state, err = camera.restart()
                    if err:
                        raise err
                continue
            
            image_height, image_width = color_image.shape[:2]
            timestamp = time.time()
            
            # 更新 IMU 数据
            if imu_data:
                collector.update_imu(imu_data)
            
            # 姿态检测
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            result = estimator.detect(image_rgb, timestamp=timestamp)
            
            # 稳定性检测
            if result is not None:
                stability_detector.add_frame(result)
            stability = stability_detector.get_stability()
            stable_progress = min(1.0, stability.stable_frames / float(collector_config.required_stable_frames))
            
            # 测量计算
            if measurement_engine and result.pose.detected:
                if (timestamp - last_measurement_ts) >= 0.2:
                    if camera.is_realsense and depth_image is not None:
                        measurement_values_cm = measurement_engine.calculate_measurements(
                            result, depth_image, image_width, image_height, timestamp
                        )
                    else:
                        measurement_values_cm = measurement_engine.calculate_measurements_from_world_landmarks(
                            result, timestamp
                        )
                    last_measurement_ts = timestamp
            
            # 计算 FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # 绘制
            annotated_image = color_image.copy()
            draw_skeleton(annotated_image, result, image_width, image_height)
            
            # 信息面板
            cv2.rectangle(annotated_image, (0, 0), (340, 128), (0, 0, 0), -1)
            annotated_image = draw_chinese_text(annotated_image, "后端: MediaPipe", 10, 25, 16, (0, 255, 255))
            annotated_image = draw_chinese_text(annotated_image, f"帧率: {fps:.1f}", 10, 50, 16, (0, 255, 0))
            annotated_image = draw_chinese_text(annotated_image, "身体: 33点, 手部: 21点x2", 10, 75, 14, (200, 200, 200))

            # 进度条
            draw_progress_bar(annotated_image, stable_progress, 10, 95, 220, 16)
            annotated_image = draw_chinese_text(annotated_image,
                f"稳定: {stability.stable_frames}/{collector_config.required_stable_frames}",
                250, 108, 12, (200, 200, 200))

            # 水平仪（仅 RealSense + IMU）
            if camera.has_imu:
                level_x = image_width - 60
                level_y = 70
                is_level = draw_level_indicator(
                    annotated_image,
                    collector.state.camera_pitch,
                    collector.state.camera_roll,
                    level_x, level_y,
                    radius=40,
                    threshold=collector_config.level_threshold
                )

                if is_level:
                    annotated_image = draw_chinese_text(annotated_image, "水平", level_x - 30, level_y + 55, 14, (0, 255, 0))
                else:
                    annotated_image = draw_chinese_text(annotated_image, "倾斜", level_x - 22, level_y + 55, 14, (0, 165, 255))

            # 状态消息
            state_msg = collector.get_state_message(
                timestamp, camera.has_imu, camera_state.mode, stable_progress
            )
            if state_msg:
                annotated_image = draw_chinese_text(annotated_image, state_msg, 10, 125, 14, (255, 255, 255))

            # 保存提示
            if collector.state.saved_hint_until > timestamp:
                annotated_image = draw_chinese_text(annotated_image, "已保存!", 260, 125, 16, (0, 255, 0))

            # 检测状态
            status_y = image_height - 30
            body_status = "身体: ✓" if result.pose.detected else "身体: ✗"
            left_status = "左: ✓" if result.left_hand and result.left_hand.detected else "左: ✗"
            right_status = "右: ✓" if result.right_hand and result.right_hand.detected else "右: ✗"

            annotated_image = draw_chinese_text(annotated_image,
                f"{body_status}  {left_status}  {right_status}",
                10, status_y, 14, (255, 255, 255))
            annotated_image = draw_chinese_text(annotated_image, "q:退出 r:重启", image_width - 160, status_y, 12, (150, 150, 150))

            # 测量面板（包含置信度信息）
            constraints_result = measurement_engine.last_constraints_result if measurement_engine else None
            panel = render_measurement_panel(image_height, panel_w, measurement_values_cm, constraints_result)
            display_frame = np.hstack((annotated_image, panel))
            
            cv2.imshow(window_name, display_frame)
            
            # 更新采集状态机
            save_result = collector.update(
                timestamp=timestamp,
                is_stable=stability.is_stable,
                stable_progress=stable_progress,
                pose_detected=result.pose.detected,
                measurement_values_cm=measurement_values_cm,
                has_imu=camera.has_imu,
                camera_mode=camera_state.mode
            )
            
            # 保存图像
            if save_result:
                cv2.imwrite(save_result["img_path"], display_frame)
                collector.state.frozen_display_frame = display_frame.copy()
                print(f"✅ 已保存: {save_result['json_path']}")
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                camera_state, err = camera.restart()
                if err:
                    raise err
                stability_detector.reset()
                collector.reset()
                measurement_values_cm = None
                if camera_state.transformer:
                    if camera.is_realsense:
                        if depth_processor:
                            depth_processor.reset()
                        else:
                            depth_processor_config = DepthProcessorConfig(
                                enable_enhancement=False,
                                enable_hole_filling=False
                            )
                            depth_processor = DepthProcessor(depth_processor_config)

                        if keypoint_stabilizer:
                            keypoint_stabilizer.reset()
                        else:
                            stabilizer_config = StabilizerConfig()
                            keypoint_stabilizer = KeypointStabilizer(stabilizer_config)

                        if not biomechanical_constraints:
                            constraints_config = ConstraintsConfig()
                            biomechanical_constraints = BiomechanicalConstraints(constraints_config)

                        measurement_engine = MeasurementEngine(
                            camera_state.transformer,
                            camera_state.hand_transformer,
                            depth_processor,
                            keypoint_stabilizer,
                            biomechanical_constraints
                        )
                    else:
                        measurement_engine = MeasurementEngine(
                            camera_state.transformer,
                            depth_processor=None,
                            keypoint_stabilizer=None,
                            biomechanical_constraints=None
                        )
            elif key == ord('c'):
                collector.manual_trigger(
                    timestamp=timestamp,
                    is_stable=stability.is_stable,
                    stable_progress=stable_progress,
                    pose_detected=result.pose.detected,
                    measurement_values_cm=measurement_values_cm,
                    has_imu=camera.has_imu,
                    camera_mode=camera_state.mode
                )

        # 清理
        estimator.close()
        camera.stop()
        cv2.destroyAllWindows()
        
        print("\n✅ 程序结束")
        return 0
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
