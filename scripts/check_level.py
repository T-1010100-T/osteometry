"""
相机水平检测器

使用加速度计检测相机是否水平放置
静止时加速度计测量重力方向：
- 完全水平: X≈0, Y≈-9.8, Z≈0 (相机正面朝前)
- 或: X≈0, Y≈0, Z≈-9.8 (取决于相机朝向)
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math

print("=" * 50)
print("RealSense D455 水平检测器")
print("=" * 50)

# 启动 IMU
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    pipeline.start(config)
    print("✅ 相机启动成功")
except Exception as e:
    print(f"❌ 启动失败: {e}")
    exit(1)

cv2.namedWindow("Level Check", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Level Check", 800, 600)

# 重力加速度
GRAVITY = 9.81
# 水平阈值（度）
LEVEL_THRESHOLD = 2.0

accel_data = {'x': 0, 'y': 0, 'z': 0}

print("\n按 'q' 退出")
print("将相机放平，观察水平指示器\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        # 获取加速度数据
        for frame in frames:
            if frame.is_motion_frame():
                motion = frame.as_motion_frame()
                if frame.get_profile().stream_type() == rs.stream.accel:
                    data = motion.get_motion_data()
                    # 简单低通滤波
                    alpha = 0.3
                    accel_data['x'] = alpha * data.x + (1-alpha) * accel_data['x']
                    accel_data['y'] = alpha * data.y + (1-alpha) * accel_data['y']
                    accel_data['z'] = alpha * data.z + (1-alpha) * accel_data['z']
        
        # 获取彩色图像
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        image = np.asanyarray(color_frame.get_data())
        h, w = image.shape[:2]
        
        # 计算倾斜角度
        ax, ay, az = accel_data['x'], accel_data['y'], accel_data['z']
        
        # 计算 pitch（前后倾斜）和 roll（左右倾斜）
        # 假设相机正常放置时 Y 轴朝下（重力方向）
        pitch = math.degrees(math.atan2(ax, math.sqrt(ay*ay + az*az)))
        roll = math.degrees(math.atan2(az, math.sqrt(ax*ax + ay*ay)))
        
        # 判断是否水平
        is_level = abs(pitch) < LEVEL_THRESHOLD and abs(roll) < LEVEL_THRESHOLD
        
        # 绘制信息面板
        panel_h = 150
        cv2.rectangle(image, (0, 0), (w, panel_h), (0, 0, 0), -1)
        
        # 标题
        cv2.putText(image, "Camera Level Check", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 加速度数据
        cv2.putText(image, f"Accel: X={ax:+.2f} Y={ay:+.2f} Z={az:+.2f} m/s2", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 倾斜角度
        pitch_color = (0, 255, 0) if abs(pitch) < LEVEL_THRESHOLD else (0, 0, 255)
        roll_color = (0, 255, 0) if abs(roll) < LEVEL_THRESHOLD else (0, 0, 255)
        
        cv2.putText(image, f"Pitch (front/back): {pitch:+.1f} deg", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 2)
        cv2.putText(image, f"Roll (left/right): {roll:+.1f} deg", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, roll_color, 2)
        
        # 水平状态
        if is_level:
            cv2.putText(image, "LEVEL OK", (w - 200, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            cv2.putText(image, "NOT LEVEL", (w - 220, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # 绘制水平仪（圆形）
        center_x, center_y = w - 100, panel_h + 100
        radius = 60
        
        # 外圈
        cv2.circle(image, (center_x, center_y), radius, (100, 100, 100), 2)
        cv2.circle(image, (center_x, center_y), radius // 3, (100, 100, 100), 1)
        
        # 气泡位置（根据倾斜角度）
        bubble_x = int(center_x + roll * 3)  # 放大显示
        bubble_y = int(center_y + pitch * 3)
        
        # 限制在圆内
        dist = math.sqrt((bubble_x - center_x)**2 + (bubble_y - center_y)**2)
        if dist > radius - 10:
            scale = (radius - 10) / dist
            bubble_x = int(center_x + (bubble_x - center_x) * scale)
            bubble_y = int(center_y + (bubble_y - center_y) * scale)
        
        bubble_color = (0, 255, 0) if is_level else (0, 165, 255)
        cv2.circle(image, (bubble_x, bubble_y), 15, bubble_color, -1)
        
        # 十字线
        cv2.line(image, (center_x - radius, center_y), (center_x + radius, center_y), (100, 100, 100), 1)
        cv2.line(image, (center_x, center_y - radius), (center_x, center_y + radius), (100, 100, 100), 1)
        
        cv2.imshow("Level Check", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\n已退出")
