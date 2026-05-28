"""
检查 RealSense D455 IMU（陀螺仪+加速度计）
"""
import pyrealsense2 as rs
import time

print("=" * 50)
print("RealSense D455 IMU 检测")
print("=" * 50)

# 查看设备信息
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("❌ 未检测到 RealSense 设备")
    exit(1)

dev = devices[0]
print(f"\n设备: {dev.get_info(rs.camera_info.name)}")
print(f"序列号: {dev.get_info(rs.camera_info.serial_number)}")
print(f"固件: {dev.get_info(rs.camera_info.firmware_version)}")

# 检查传感器
print("\n传感器列表:")
has_gyro = False
has_accel = False

for sensor in dev.sensors:
    name = sensor.get_info(rs.camera_info.name)
    print(f"  - {name}")
    
    for profile in sensor.get_stream_profiles():
        if profile.stream_type() == rs.stream.gyro:
            has_gyro = True
        if profile.stream_type() == rs.stream.accel:
            has_accel = True

print(f"\n陀螺仪 (Gyro): {'✅ 支持' if has_gyro else '❌ 不支持'}")
print(f"加速度计 (Accel): {'✅ 支持' if has_accel else '❌ 不支持'}")

if not (has_gyro and has_accel):
    print("\n该设备不支持 IMU")
    exit(0)

# 读取 IMU 数据
print("\n" + "=" * 50)
print("实时 IMU 数据 (按 Ctrl+C 退出)")
print("=" * 50)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)

try:
    pipeline.start(config)
    print("\n✅ IMU 数据流已启动\n")
    
    while True:
        frames = pipeline.wait_for_frames()
        
        for frame in frames:
            if frame.is_motion_frame():
                motion = frame.as_motion_frame()
                data = motion.get_motion_data()
                
                if frame.get_profile().stream_type() == rs.stream.gyro:
                    print(f"陀螺仪: X={data.x:+8.4f}  Y={data.y:+8.4f}  Z={data.z:+8.4f} rad/s", end="\r")
                # elif frame.get_profile().stream_type() == rs.stream.accel:
                #     print(f"加速度: X={data.x:+8.4f}  Y={data.y:+8.4f}  Z={data.z:+8.4f} m/s²")
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n已停止")
finally:
    pipeline.stop()
