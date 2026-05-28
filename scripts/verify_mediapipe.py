"""
MediaPipe 姿态估计验证脚本
测试 MediaPipe Pose 是否能正常工作
包含中文路径问题的处理
"""
import sys
import os

# MediaPipe 中文路径问题修复 - 必须在导入mediapipe之前
os.makedirs('D:/mediapipe_cache', exist_ok=True)
os.environ['MEDIAPIPE_CACHE_DIR'] = 'D:/mediapipe_cache'
os.environ['XDG_CACHE_HOME'] = 'D:/mediapipe_cache'
os.environ['GLOG_log_dir'] = 'D:/mediapipe_cache'


def setup_chinese_path_workaround():
    """
    处理中文路径问题
    MediaPipe 的 C++ 层不支持中文路径
    """
    # 检查当前路径是否包含非 ASCII 字符
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    try:
        current_path.encode('ascii')
        return True  # 路径是纯 ASCII，无需处理
    except UnicodeEncodeError:
        pass  # 路径包含非 ASCII 字符
    
    print("⚠️ 检测到路径包含中文字符")
    
    # 检查是否已设置 Junction
    junction_path = os.environ.get('MEDIAPIPE_JUNCTION_PATH')
    if junction_path and os.path.exists(junction_path):
        print(f"  ✅ 使用 Junction 路径: {junction_path}")
        return True
    
    print("  💡 解决方案:")
    print("  1. 以管理员身份运行 PowerShell")
    print("  2. 执行以下命令创建 Junction:")
    print()
    
    # 获取 site-packages 路径
    import site
    site_packages = site.getsitepackages()[0]
    print(f'     New-Item -ItemType Junction -Path "C:\\mediapipe_temp" -Target "{site_packages}"')
    print()
    print("  3. 设置环境变量:")
    print('     $env:MEDIAPIPE_JUNCTION_PATH = "C:\\mediapipe_temp"')
    print()
    
    return False


def test_mediapipe_import():
    """测试 MediaPipe 导入"""
    print("[1] 测试 MediaPipe 导入...")
    
    try:
        import mediapipe as mp
        print(f"  ✅ MediaPipe 版本: {mp.__version__}")
        return True
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_pose_model():
    """测试 Pose 模型加载"""
    print("\n[2] 测试 Pose 模型...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        mp_pose = mp.solutions.pose
        
        # 测试不同复杂度的模型
        for complexity in [0, 1, 2]:
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=complexity,
                min_detection_confidence=0.5
            )
            pose.close()
            print(f"  ✅ 模型复杂度 {complexity} 加载成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return False


def test_pose_detection():
    """测试姿态检测"""
    print("\n[3] 测试姿态检测...")
    
    try:
        import mediapipe as mp
        import numpy as np
        import cv2
        
        # 创建测试图像（黑色背景）
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        
        # 运行检测
        results = pose.process(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            print(f"  ✅ 检测到 {len(results.pose_landmarks.landmark)} 个关键点")
        else:
            print("  ✅ 检测运行正常（测试图像中无人体）")
        
        pose.close()
        return True
    except Exception as e:
        print(f"  ❌ 检测失败: {e}")
        return False


def test_with_camera():
    """使用相机测试实时姿态检测"""
    print("\n[4] 实时姿态检测测试...")
    
    try:
        import mediapipe as mp
        import cv2
        
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        # 尝试使用 RealSense
        try:
            import pyrealsense2 as rs
            use_realsense = True
            print("  使用 RealSense 相机")
        except ImportError:
            use_realsense = False
            print("  使用默认摄像头")
        
        if use_realsense:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
        else:
            cap = cv2.VideoCapture(0)
        
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("  按 'q' 退出")
        
        frame_count = 0
        detected_count = 0
        
        while frame_count < 300:  # 最多测试 300 帧（约10秒）
            if use_realsense:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data())
            else:
                ret, image = cap.read()
                if not ret:
                    continue
            
            # 检测姿态
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                detected_count += 1
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
            
            # 显示信息
            cv2.putText(image, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Detected: {detected_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Press 'q' to exit", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('MediaPipe Pose Test', image)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 清理
        pose.close()
        cv2.destroyAllWindows()
        
        if use_realsense:
            pipeline.stop()
        else:
            cap.release()
        
        print(f"  ✅ 测试完成: {detected_count}/{frame_count} 帧检测到人体")
        return True
        
    except Exception as e:
        print(f"  ❌ 实时检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("MediaPipe 姿态估计验证")
    print("=" * 50)
    
    # 0. 检查中文路径
    print("\n[0] 检查路径...")
    if not setup_chinese_path_workaround():
        print("\n⚠️ 请先解决中文路径问题，然后重新运行此脚本")
        return 1
    
    results = []
    
    # 1. 导入测试
    results.append(test_mediapipe_import())
    if not results[-1]:
        return 1
    
    # 2. 模型加载
    results.append(test_pose_model())
    
    # 3. 检测测试
    results.append(test_pose_detection())
    
    # 4. 实时测试
    response = input("\n是否进行实时检测测试? (y/n): ")
    if response.lower() == 'y':
        # 需要导入 numpy（在函数内使用）
        import numpy as np
        results.append(test_with_camera())
    
    # 总结
    print("\n" + "=" * 50)
    if all(results):
        print("✅ MediaPipe 验证完成！")
    else:
        print("⚠️ 部分测试失败")
    print("=" * 50)
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
