"""
骨骼段测量采集器
实时显示详细的骨骼段测量数据
"""
import sys
import time
import os
import json
from datetime import datetime
import threading
import queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_default_junction_path = r"C:\mediapipe_temp"
_junction_path = os.environ.get("MEDIAPIPE_JUNCTION_PATH") or _default_junction_path
if _junction_path and os.path.exists(_junction_path):
    if "MEDIAPIPE_JUNCTION_PATH" not in os.environ:
        os.environ["MEDIAPIPE_JUNCTION_PATH"] = _junction_path
    if _junction_path not in sys.path:
        sys.path.insert(0, _junction_path)

try:
    import pyrealsense2 as rs
    import numpy as np
    import cv2
except ImportError as e:
    print(f"缺少依赖: {e}")
    sys.exit(1)

from src.core import HolisticEstimator
from src.core.coordinate_transformer import CoordinateTransformer, calculate_distance
from src.core.skeleton import Skeleton3D, Point3D
from src.core.keypoint_filter import KeypointFilter
from src.measurement.measurement_engine import MeasurementEngine
from src.hardware.frame_set import Intrinsics

MP_IMAGE_SCALE = 0.5


def calculate_bone_segments(skeleton: Skeleton3D):
    """计算所有骨骼段长度"""
    segments = {}
    
    # 获取关键点
    nose = skeleton.get_joint('nose')
    left_shoulder = skeleton.get_joint('left_shoulder')
    right_shoulder = skeleton.get_joint('right_shoulder')
    left_elbow = skeleton.get_joint('left_elbow')
    right_elbow = skeleton.get_joint('right_elbow')
    left_wrist = skeleton.get_joint('left_wrist')
    right_wrist = skeleton.get_joint('right_wrist')
    left_pinky = skeleton.get_joint('left_pinky')
    right_pinky = skeleton.get_joint('right_pinky')
    left_hip = skeleton.get_joint('left_hip')
    right_hip = skeleton.get_joint('right_hip')
    left_knee = skeleton.get_joint('left_knee')
    right_knee = skeleton.get_joint('right_knee')
    left_ankle = skeleton.get_joint('left_ankle')
    right_ankle = skeleton.get_joint('right_ankle')
    left_foot_index = skeleton.get_joint('left_foot_index')
    right_foot_index = skeleton.get_joint('right_foot_index')
    
    # 计算肩部中心和髋部中心
    if left_shoulder and right_shoulder:
        shoulder_center = Point3D(
            x=(left_shoulder.x + right_shoulder.x) / 2,
            y=(left_shoulder.y + right_shoulder.y) / 2,
            z=(left_shoulder.z + right_shoulder.z) / 2
        )
    else:
        shoulder_center = None
    
    if left_hip and right_hip:
        hip_center = Point3D(
            x=(left_hip.x + right_hip.x) / 2,
            y=(left_hip.y + right_hip.y) / 2,
            z=(left_hip.z + right_hip.z) / 2
        )
    else:
        hip_center = None
    
    # 计算脊柱中部（髋部中心和肩部中心之间的58%位置）
    if hip_center and shoulder_center:
        spine_mid = Point3D(
            x=hip_center.x + (shoulder_center.x - hip_center.x) * 0.58,
            y=hip_center.y + (shoulder_center.y - hip_center.y) * 0.58,
            z=hip_center.z + (shoulder_center.z - hip_center.z) * 0.58
        )
    else:
        spine_mid = None
    
    # 脊柱段
    if hip_center and spine_mid:
        segments['脊柱底部到中部'] = calculate_distance(hip_center, spine_mid) * 1000
    if spine_mid and shoulder_center:
        segments['脊柱中部到肩部'] = calculate_distance(spine_mid, shoulder_center) * 1000
    if shoulder_center and nose:
        segments['脊柱肩部到头部'] = calculate_distance(shoulder_center, nose) * 1000
    
    # 左臂
    if shoulder_center and left_shoulder:
        segments['脊柱肩部到左肩'] = calculate_distance(shoulder_center, left_shoulder) * 1000
    if left_shoulder and left_elbow:
        segments['左肩到左肘'] = calculate_distance(left_shoulder, left_elbow) * 1000
    if left_elbow and left_wrist:
        segments['左肘到左腕'] = calculate_distance(left_elbow, left_wrist) * 1000
    if left_wrist and left_pinky:
        segments['左腕到左手'] = calculate_distance(left_wrist, left_pinky) * 1000
    
    # 右臂
    if shoulder_center and right_shoulder:
        segments['脊柱肩部到右肩'] = calculate_distance(shoulder_center, right_shoulder) * 1000
    if right_shoulder and right_elbow:
        segments['右肩到右肘'] = calculate_distance(right_shoulder, right_elbow) * 1000
    if right_elbow and right_wrist:
        segments['右肘到右腕'] = calculate_distance(right_elbow, right_wrist) * 1000
    if right_wrist and right_pinky:
        segments['右腕到右手'] = calculate_distance(right_wrist, right_pinky) * 1000
    
    # 左腿
    if hip_center and left_hip:
        segments['脊柱底部到左髋'] = calculate_distance(hip_center, left_hip) * 1000
    if left_hip and left_knee:
        segments['左髋到左膝'] = calculate_distance(left_hip, left_knee) * 1000
    if left_knee and left_ankle:
        segments['左膝到左踝'] = calculate_distance(left_knee, left_ankle) * 1000
    if left_ankle and left_foot_index:
        segments['左踝到左脚'] = calculate_distance(left_ankle, left_foot_index) * 1000
    
    # 右腿
    if hip_center and right_hip:
        segments['脊柱底部到右髋'] = calculate_distance(hip_center, right_hip) * 1000
    if right_hip and right_knee:
        segments['右髋到右膝'] = calculate_distance(right_hip, right_knee) * 1000
    if right_knee and right_ankle:
        segments['右膝到右踝'] = calculate_distance(right_knee, right_ankle) * 1000
    if right_ankle and right_foot_index:
        segments['右踝到右脚'] = calculate_distance(right_ankle, right_foot_index) * 1000
    
    # 计算身高（累加法）
    height = 0
    if '脊柱底部到中部' in segments:
        height += segments['脊柱底部到中部']
    if '脊柱中部到肩部' in segments:
        height += segments['脊柱中部到肩部']
    if '脊柱肩部到头部' in segments:
        height += segments['脊柱肩部到头部']
    
    # 添加腿长（取左右平均）
    left_leg = 0
    if '左髋到左膝' in segments:
        left_leg += segments['左髋到左膝']
    if '左膝到左踝' in segments:
        left_leg += segments['左膝到左踝']
    if '左踝到左脚' in segments:
        left_leg += segments['左踝到左脚']
    
    right_leg = 0
    if '右髋到右膝' in segments:
        right_leg += segments['右髋到右膝']
    if '右膝到右踝' in segments:
        right_leg += segments['右膝到右踝']
    if '右踝到右脚' in segments:
        right_leg += segments['右踝到右脚']
    
    if left_leg > 0 and right_leg > 0:
        height += (left_leg + right_leg) / 2
        height += 120  # 头顶高度
        height += 70   # 脚底高度
    
    segments['身高'] = height
    
    return segments


def draw_bone_segments(image, segments, start_x=10, start_y=30):
    """在图像上绘制骨骼段测量数据"""
    y = start_y
    line_height = 20
    
    # 标题
    cv2.putText(image, "Bone Segment Measurements (mm)", (start_x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += 30
    
    # 定义显示顺序
    display_order = [
        '身高',
        '脊柱底部到中部',
        '脊柱中部到肩部',
        '脊柱肩部到头部',
        '脊柱肩部到左肩',
        '左肩到左肘',
        '左肘到左腕',
        '左腕到左手',
        '脊柱肩部到右肩',
        '右肩到右肘',
        '右肘到右腕',
        '右腕到右手',
        '脊柱底部到左髋',
        '左髋到左膝',
        '左膝到左踝',
        '左踝到左脚',
        '脊柱底部到右髋',
        '右髋到右膝',
        '右膝到右踝',
        '右踝到右脚',
    ]
    
    for name in display_order:
        if name in segments:
            value = segments[name]
            if name == '身高':
                text = f"{name}: {value:.1f}"
                color = (0, 255, 0)
            else:
                text = f"{name}: {value:.1f}"
                color = (255, 255, 255)
        else:
            text = f"{name}: --"
            color = (128, 128, 128)
        
        cv2.putText(image, text, (start_x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += line_height
        
        # 换列
        if y > image.shape[0] - 30:
            start_x += 250
            y = start_y + 30
    
    return image


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.latest_segments = {}
        self.running = True
        self.fps = 0


def camera_thread(pipeline, align, shared_state):
    """摄像头采集线程"""
    while shared_state.running:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            with shared_state.lock:
                shared_state.latest_color_frame = color_image.copy()
                shared_state.latest_depth_frame = depth_image.copy()
        except Exception as e:
            if shared_state.running:
                print(f"摄像头错误: {e}")
            time.sleep(0.01)


def processing_thread(shared_state, holistic_estimator, transformer, keypoint_filter, intrinsics):
    """处理线程"""
    fps_counter = 0
    fps_start = time.time()
    
    while shared_state.running:
        try:
            with shared_state.lock:
                color_image = shared_state.latest_color_frame
                depth_image = shared_state.latest_depth_frame
            
            if color_image is None or depth_image is None:
                time.sleep(0.01)
                continue
            
            # MediaPipe检测
            if MP_IMAGE_SCALE and MP_IMAGE_SCALE != 1.0:
                mp_bgr = cv2.resize(color_image, (0, 0), fx=MP_IMAGE_SCALE, fy=MP_IMAGE_SCALE)
            else:
                mp_bgr = color_image
            
            image_rgb = cv2.cvtColor(mp_bgr, cv2.COLOR_BGR2RGB)
            holistic_result = holistic_estimator.detect(image_rgb, timestamp=time.time())
            
            # 测量骨骼段
            if holistic_result.pose.detected:
                try:
                    pose_result = holistic_result.to_pose_result()
                    points_3d = transformer.transform_with_filter(
                        pose_result, depth_image,
                        min_visibility=0.5, depth_range=(0.3, 3.0)
                    )
                    points_3d_filtered = keypoint_filter.update(points_3d)
                    skeleton = Skeleton3D.from_points(points_3d_filtered)
                    segments = calculate_bone_segments(skeleton)
                    
                    with shared_state.lock:
                        shared_state.latest_segments = segments
                except Exception:
                    pass
            
            # 计算FPS
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                with shared_state.lock:
                    shared_state.fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
                
        except Exception as e:
            if shared_state.running:
                print(f"处理错误: {e}")


def main():
    print("=" * 50)
    print("骨骼段测量采集器 - 多线程版")
    print("=" * 50)
    print("按 'q' 退出")
    print("按 's' 保存当前数据")
    print()
    
    # 初始化摄像头
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
    print("✅ 摄像头启动成功!")
    
    # 初始化组件
    holistic_estimator = HolisticEstimator()
    camera_intrinsics = Intrinsics(
        width=intrinsics.width, height=intrinsics.height,
        fx=intrinsics.fx, fy=intrinsics.fy,
        ppx=intrinsics.ppx, ppy=intrinsics.ppy
    )
    transformer = CoordinateTransformer(camera_intrinsics)
    keypoint_filter = KeypointFilter(window_size=5)
    
    # 共享状态
    shared_state = SharedState()
    
    # 启动后台线程
    cam_thread = threading.Thread(
        target=camera_thread,
        args=(pipeline, align, shared_state),
        daemon=True
    )
    proc_thread = threading.Thread(
        target=processing_thread,
        args=(shared_state, holistic_estimator, transformer, keypoint_filter, intrinsics),
        daemon=True
    )
    
    cam_thread.start()
    proc_thread.start()
    print("✅ 多线程启动成功!")
    
    # 创建窗口
    cv2.namedWindow('Bone Segment Collector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Bone Segment Collector', 1280, 480)
    
    print("✅ 初始化完成")
    
    try:
        while True:
            # 获取最新数据
            with shared_state.lock:
                color_image = shared_state.latest_color_frame
                segments = shared_state.latest_segments.copy() if shared_state.latest_segments else {}
                current_fps = shared_state.fps
            
            # 创建显示画面
            display = np.zeros((480, 1280, 3), dtype=np.uint8)
            
            # 左侧：摄像头画面
            if color_image is not None:
                display[:, :640] = color_image
            else:
                cv2.putText(display, "Waiting for camera...", (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 右侧：测量数据
            cv2.rectangle(display, (640, 0), (1280, 480), (30, 30, 30), -1)
            
            # FPS
            cv2.putText(display, f"FPS: {current_fps}", (650, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 绘制测量数据
            draw_bone_segments(display, segments, start_x=650, start_y=50)
            
            # 提示
            cv2.putText(display, "q:quit s:save", (10, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Bone Segment Collector', display)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n退出...")
                break
            elif key == ord('s'):
                # 保存数据
                save_dir = "data/sessions"
                os.makedirs(save_dir, exist_ok=True)
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                
                # 保存JSON
                json_path = os.path.join(save_dir, f"bone_segments_{timestamp_str}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': timestamp_str,
                        'segments': segments
                    }, f, indent=2, ensure_ascii=False)
                print(f"✅ 数据已保存: {json_path}")
                
                # 保存截图
                img_path = os.path.join(save_dir, f"bone_segments_{timestamp_str}.jpg")
                cv2.imwrite(img_path, display)
                print(f"✅ 截图已保存: {img_path}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shared_state.running = False
        cam_thread.join(timeout=2)
        proc_thread.join(timeout=2)
        holistic_estimator.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("✅ 程序结束")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
