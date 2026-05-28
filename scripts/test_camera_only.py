"""
测试摄像头显示 - 不包含MediaPipe处理
用于诊断UI卡顿问题
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyrealsense2 as rs
import numpy as np
import cv2
import time

print("=" * 50)
print("摄像头测试 - 纯显示模式")
print("=" * 50)
print("按 'q' 退出, 按 'r' 重新启动摄像头")
print()

# 初始化 RealSense
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)
colorizer = rs.colorizer()


def start_pipeline():
    pipeline = rs.pipeline()
    try:
        pipeline.start(config)
        print("[OK] Camera started")
        return pipeline
    except RuntimeError as e:
        print(f"[ERR] Camera start failed: {e}")
        try:
            pipeline.stop()
        except Exception:
            pass
        raise


def stop_pipeline(pipeline):
    try:
        pipeline.stop()
    except Exception:
        pass

try:
    pipeline = start_pipeline()
except RuntimeError as e:
    print(f"[ERR] Camera start failed: {e}")
    input("按回车键退出...")
    sys.exit(1)

# 创建窗口
cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Test', 640, 480)

fps_counter = 0
fps_start_time = time.time()
current_fps = 0
missing_counter = 0

print("[OK] Starting preview...")

try:
    while True:
        # 读取帧
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
        except Exception as e:
            print(f"[ERR] wait_for_frames failed: {e}")
            color_frame = None
            depth_frame = None

        # 无论是否拿到帧，都要 pump 一次 OpenCV 事件，避免窗口“未响应”
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n退出...")
            break
        if key == ord('r'):
            print("\n[OK] Restarting camera...")
            stop_pipeline(pipeline)
            time.sleep(0.2)
            pipeline = start_pipeline()
            fps_counter = 0
            fps_start_time = time.time()
            current_fps = 0
            missing_counter = 0
            continue

        if not color_frame or not depth_frame:
            missing_counter += 1
            placeholder = np.zeros((480, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No frames", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(placeholder, "Press 'q' to quit | 'r' to restart", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Camera Test', placeholder)

            # 连续拿不到帧时，提示你手动重启；也可以在这里改成自动重启
            if missing_counter % 30 == 0:
                print(f"[ERR] No frames for {missing_counter} loops. Press 'r' to restart.")
            continue
        else:
            missing_counter = 0
        
        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        
        # 计算FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # 并排显示 RGB + Depth
        images = np.hstack((color_image, depth_colormap))

        # 绘制FPS
        cv2.putText(images, f"FPS: {current_fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(images, "Press 'q' to quit | 'r' to restart", (10, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(images, "RGB", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(images, "Depth", (650, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 显示
        cv2.imshow('Camera Test', images)

except KeyboardInterrupt:
    print("\n用户中断")
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    stop_pipeline(pipeline)
    cv2.destroyAllWindows()
    print("[OK] Program finished")
