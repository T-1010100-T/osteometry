"""
骨骼节点长度测试脚本

实时测量并显示所有骨骼段的长度

按 'q' 退出
按 's' 保存当前测量数据
"""
import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pyrealsense2 as rs
    import numpy as np
    import cv2
except ImportError as e:
    print(f"缺少依赖: {e}")
    sys.exit(1)

from src.core import HolisticEstimator, POSE_CONNECTIONS, HAND_CONNECTIONS
from src.core.coordinate_transformer import CoordinateTransformer, calculate_distance
from src.core.skeleton import Skeleton3D
from src.core.hand_skeleton import HandSkeleton3D
from src.core.hand_coordinate_transformer import HandCoordinateTransformer
from src.core.constants import LANDMARK_NAMES, HAND_LANDMARK_NAMES
from src.hardware.frame_set import Intrinsics


# 身体骨骼段定义 (名称, 起点索引, 终点索引)
BODY_BONES = [
    # 躯干
    ("肩宽", 11, 12),           # left_shoulder - right_shoulder
    ("左躯干", 11, 23),         # left_shoulder - left_hip
    ("右躯干", 12, 24),         # right_shoulder - right_hip
    ("髋宽", 23, 24),           # left_hip - right_hip
    
    # 左臂
    ("左上臂", 11, 13),         # left_shoulder - left_elbow
    ("左前臂", 13, 15),         # left_elbow - left_wrist
    ("左腕-小指", 15, 17),      # left_wrist - left_pinky
    ("左腕-食指", 15, 19),      # left_wrist - left_index
    ("左腕-拇指", 15, 21),      # left_wrist - left_thumb
    
    # 右臂
    ("右上臂", 12, 14),         # right_shoulder - right_elbow
    ("右前臂", 14, 16),         # right_elbow - right_wrist
    ("右腕-小指", 16, 18),      # right_wrist - right_pinky
    ("右腕-食指", 16, 20),      # right_wrist - right_index
    ("右腕-拇指", 16, 22),      # right_wrist - right_thumb
    
    # 左腿
    ("左大腿", 23, 25),         # left_hip - left_knee
    ("左小腿", 25, 27),         # left_knee - left_ankle
    ("左脚踝-脚跟", 27, 29),    # left_ankle - left_heel
    ("左脚踝-脚趾", 27, 31),    # left_ankle - left_foot_index
    
    # 右腿
    ("右大腿", 24, 26),         # right_hip - right_knee
    ("右小腿", 26, 28),         # right_knee - right_ankle
    ("右脚踝-脚跟", 28, 30),    # right_ankle - right_heel
    ("右脚踝-脚趾", 28, 32),    # right_ankle - right_foot_index
    
    # 头部/颈部
    ("鼻-左眼内", 0, 1),
    ("鼻-右眼内", 0, 4),
    ("左眼内-左眼", 1, 2),
    ("左眼-左眼外", 2, 3),
    ("左眼外-左耳", 3, 7),
    ("右眼内-右眼", 4, 5),
    ("右眼-右眼外", 5, 6),
    ("右眼外-右耳", 6, 8),
    ("嘴左-嘴右", 9, 10),
]


def measure_bone_lengths(points_3d, bones):
    """测量骨骼段长度"""
    results = {}
    for name, start_idx, end_idx in bones:
        if start_idx < len(points_3d) and end_idx < len(points_3d):
            p1 = points_3d[start_idx]
            p2 = points_3d[end_idx]
            if p1 and p2 and p1.is_valid() and p2.is_valid():
                dist = calculate_distance(p1, p2)
                results[name] = dist * 100  # 转换为厘米
            else:
                results[name] = None
        else:
            results[name] = None
    return results


def draw_measurements(image, measurements, start_x, start_y, title="骨骼长度"):
    """在图像上绘制测量结果"""
    cv2.putText(image, title, (start_x, start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    y = start_y + 25
    for name, value in measurements.items():
        if value is not None:
            text = f"{name}: {value:.1f}cm"
            color = (0, 255, 0)
        else:
            text = f"{name}: --"
            color = (128, 128, 128)
        
        cv2.putText(image, text, (start_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 18
        
        if y > image.shape[0] - 20:
            start_x += 180
            y = start_y + 25
    
    return y


def main():
    print("=" * 50)
    print("骨骼节点长度测试")
    print("=" * 50)
    print()
    print("按键:")
    print("  q - 退出")
    print("  s - 保存当前测量数据")
    print()
    
    # 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    align = rs.align(rs.stream.color)
    
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"❌ 摄像头启动失败: {e}")
        input("按回车键退出...")
        return 1
    
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics = depth_stream.get_intrinsics()
    
    print(f"✅ 摄像头启动成功!")
    
    # 初始化
    holistic_estimator = HolisticEstimator()
    camera_intrinsics = Intrinsics(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        ppx=intrinsics.ppx,
        ppy=intrinsics.ppy
    )
    transformer = CoordinateTransformer(camera_intrinsics)
    hand_transformer = HandCoordinateTransformer(camera_intrinsics)
    
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    # 测量结果缓存
    current_body_measurements = {}
    current_left_hand_measurements = {}
    current_right_hand_measurements = {}
    
    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=10000)
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            image_height, image_width = color_image.shape[:2]
            timestamp = time.time()
            
            # 检测
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            holistic_result = holistic_estimator.detect(image_rgb, timestamp=timestamp)
            
            # 身体测量
            if holistic_result.pose.detected:
                try:
                    pose_result = holistic_result.to_pose_result()
                    points_3d = transformer.transform_with_filter(
                        pose_result, depth_image,
                        min_visibility=0.3, depth_range=(0.3, 4.0)
                    )
                    current_body_measurements = measure_bone_lengths(points_3d, BODY_BONES)
                except Exception as e:
                    pass
            
            # 计算 FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # 绘制
            annotated_image = color_image.copy()
            
            # 绘制骨骼
            if holistic_result.pose.detected:
                landmarks = holistic_result.pose.landmarks
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        lm1, lm2 = landmarks[start_idx], landmarks[end_idx]
                        if lm1.visibility > 0.3 and lm2.visibility > 0.3:
                            x1, y1 = int(lm1.x * image_width), int(lm1.y * image_height)
                            x2, y2 = int(lm2.x * image_width), int(lm2.y * image_height)
                            cv2.line(annotated_image, (x1, y1), (x2, y2), (200, 200, 200), 2)
                
                for i, lm in enumerate(landmarks):
                    if lm.visibility > 0.3:
                        x, y = int(lm.x * image_width), int(lm.y * image_height)
                        cv2.circle(annotated_image, (x, y), 4, (255, 200, 0), -1)
                        # 显示节点编号
                        cv2.putText(annotated_image, str(i), (x+5, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            # 绘制测量结果
            cv2.rectangle(annotated_image, (0, 0), (400, image_height), (0, 0, 0), -1)
            cv2.putText(annotated_image, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            draw_measurements(annotated_image, current_body_measurements, 10, 50, "身体骨骼长度")
            
            cv2.putText(annotated_image, "q:退出 s:保存", (10, image_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            cv2.imshow('Bone Length Test', annotated_image)
            cv2.pollKey()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存数据
                save_data = {
                    'timestamp': datetime.now().isoformat(),
                    'body_bones': current_body_measurements,
                }
                filename = f"data/sessions/bone_lengths_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                print(f"✅ 数据已保存: {filename}")
        
        holistic_estimator.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n✅ 测试结束")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
