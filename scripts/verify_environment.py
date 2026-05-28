"""
环境验证脚本 - 检查所有依赖是否正确安装
"""
import sys
import os

# MediaPipe 中文路径问题修复
os.makedirs('D:/mediapipe_cache', exist_ok=True)
os.environ['MEDIAPIPE_CACHE_DIR'] = 'D:/mediapipe_cache'
os.environ['XDG_CACHE_HOME'] = 'D:/mediapipe_cache'
os.environ['GLOG_log_dir'] = 'D:/mediapipe_cache'


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("  ✅ Python 版本符合要求")
        return True
    else:
        print("  ❌ 需要 Python 3.8+")
        return False


def check_dependencies():
    """检查依赖库"""
    dependencies = [
        ("pyrealsense2", "Intel RealSense SDK"),
        ("mediapipe", "MediaPipe"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("open3d", "Open3D"),
        ("yaml", "PyYAML"),
        ("loguru", "Loguru"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow"),
    ]
    
    all_ok = True
    print("\n依赖库检查:")
    
    for module, name in dependencies:
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "unknown")
            print(f"  ✅ {name}: {version}")
        except ImportError as e:
            print(f"  ❌ {name}: 未安装 ({e})")
            all_ok = False
    
    return all_ok


def check_realsense_device():
    """检查 RealSense 设备"""
    print("\nRealSense 设备检查:")
    
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("  ⚠️ 未检测到 RealSense 设备（请确保相机已连接）")
            return False
        
        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            firmware = dev.get_info(rs.camera_info.firmware_version)
            print(f"  ✅ 设备: {name}")
            print(f"     序列号: {serial}")
            print(f"     固件: {firmware}")
        
        return True
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False


def check_mediapipe():
    """检查 MediaPipe 是否可用"""
    print("\nMediaPipe 检查:")
    
    # 设置中文路径修复
    import os
    junction_path = r'C:\mediapipe_temp'
    if os.path.exists(junction_path):
        os.environ['MEDIAPIPE_JUNCTION_PATH'] = junction_path
        print(f"  ✅ Junction 路径已设置: {junction_path}")
    
    try:
        import mediapipe as mp
        print(f"  ✅ MediaPipe 版本: {mp.__version__}")
        
        # 尝试初始化 Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=0)
        pose.close()
        print("  ✅ Pose 模型加载成功")
        return True
    except Exception as e:
        print(f"  ❌ MediaPipe 初始化失败: {e}")
        print("  💡 提示: 如果路径包含中文，需要设置 Junction")
        return False


def main():
    print("=" * 50)
    print("RealSense 人体测量系统 - 环境验证")
    print("=" * 50)
    
    results = []
    
    # 1. Python 版本
    print("\n[1/4] Python 版本")
    results.append(("Python", check_python_version()))
    
    # 2. 依赖库
    print("\n[2/4] 依赖库")
    results.append(("依赖库", check_dependencies()))
    
    # 3. RealSense 设备
    print("\n[3/4] RealSense 设备")
    results.append(("RealSense", check_realsense_device()))
    
    # 4. MediaPipe
    print("\n[4/4] MediaPipe")
    results.append(("MediaPipe", check_mediapipe()))
    
    # 总结
    print("\n" + "=" * 50)
    print("验证结果:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有检查通过！可以开始开发。")
    else:
        print("⚠️ 部分检查失败，请根据上方提示解决问题。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
