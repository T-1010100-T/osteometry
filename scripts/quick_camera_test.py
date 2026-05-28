"""
快速摄像头测试 - 直接启动实时预览
按 'q' 退出
"""
import sys

try:
    import numpy as np
    import cv2
except ImportError as e:
    print(f"[ERR] Missing dependency: {e}")
    sys.exit(1)

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def _try_opencv_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def _run_opencv_preview():
    print("Starting OpenCV camera preview...")
    print("Press 'q' to exit")

    cap = None
    used_index = None
    for idx in range(0, 6):
        cap = _try_opencv_camera(idx)
        if cap is not None:
            used_index = idx
            break

    if cap is None:
        print("[ERR] No webcam found (index 0-5)")
        return 1

    try:
        print(f"[OK] Camera opened. index={used_index}")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERR] Failed to read frame")
                return 1

            cv2.putText(frame, f"OpenCV Camera index={used_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to exit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Camera Preview', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("[OK] Done")
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _run_realsense_preview():
    if rs is None:
        return None

    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            return None
    except Exception:
        return None

    print("Starting RealSense preview...")
    print("Press 'q' to exit")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    try:
        pipeline.start(config)
        print("[OK] Camera started")

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            images = np.hstack((color_image, depth_colormap))

            cv2.putText(images, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(images, "Depth", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(images, "Press 'q' to exit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('RealSense D455', images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("[OK] Done")
        return 0
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

def main():
    try:
        realsense_rc = _run_realsense_preview()
        if realsense_rc is not None:
            return realsense_rc

        return _run_opencv_preview()
    except Exception as e:
        print(f"[ERR] Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
