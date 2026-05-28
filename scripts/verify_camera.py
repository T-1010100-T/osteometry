"""
RealSense D455 相机验证脚本
测试相机连接、数据流获取、深度图显示
"""
import sys

try:
    import pyrealsense2 as rs
    import numpy as np
    import cv2
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请先运行: pip install -r requirements.txt")
    sys.exit(1)


def test_camera_connection():
    """测试相机连接"""
    print("[1] 检测相机连接...")
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("  ❌ 未检测到 RealSense 设备")
        print("  请检查:")
        print("    - 相机是否已连接")
        print("    - 是否使用 USB 3.0 端口")
        print("    - 驱动是否已安装")
        return None
    
    device = devices[0]
    print(f"  ✅ 检测到: {device.get_info(rs.camera_info.name)}")
    return device


def test_data_stream():
    """测试数据流获取"""
    print("\n[2] 测试数据流...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        # 启动流
        profile = pipeline.start(config)
        print("  ✅ 数据流启动成功")
        
        # 获取相机内参
        depth_stream = profile.get_stream(rs.stream.depth)
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        print(f"  ✅ 深度相机内参:")
        print(f"     分辨率: {intrinsics.width}x{intrinsics.height}")
        print(f"     焦距: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
        print(f"     主点: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
        
        # 获取几帧测试
        print("  获取测试帧...")
        for i in range(10):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                print(f"  ✅ 帧 {i+1}: RGB + Depth 获取成功")
        
        pipeline.stop()
        return True
        
    except Exception as e:
        print(f"  ❌ 数据流错误: {e}")
        return False


def test_live_preview():
    """实时预览测试"""
    print("\n[3] 实时预览测试...")
    print("  按 'q' 退出预览")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 对齐器
    align = rs.align(rs.stream.color)
    
    # 深度着色器
    colorizer = rs.colorizer()
    
    try:
        pipeline.start(config)
        
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            # 并排显示
            images = np.hstack((color_image, depth_colormap))
            
            # 添加文字说明
            cv2.putText(images, "RGB", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(images, "Depth", (650, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(images, "Press 'q' to exit", (10, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('RealSense D455 Test', images)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        pipeline.stop()
        cv2.destroyAllWindows()
        print("  ✅ 实时预览测试完成")
        return True
        
    except Exception as e:
        print(f"  ❌ 预览错误: {e}")
        return False


def main():
    print("=" * 50)
    print("RealSense D455 相机验证")
    print("=" * 50)
    
    # 1. 连接测试
    device = test_camera_connection()
    if device is None:
        return 1
    
    # 2. 数据流测试
    if not test_data_stream():
        return 1
    
    # 3. 实时预览
    response = input("\n是否进行实时预览测试? (y/n): ")
    if response.lower() == 'y':
        test_live_preview()
    
    print("\n" + "=" * 50)
    print("✅ 相机验证完成！")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
