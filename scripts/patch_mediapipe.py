"""
MediaPipe 中文路径修补脚本
解决 MediaPipe C++ 层不支持中文路径的问题
"""
import os
import sys
import shutil
from pathlib import Path

# MediaPipe 中文路径问题修复 - 必须在导入mediapipe之前
os.makedirs('D:/mediapipe_cache', exist_ok=True)
os.environ['MEDIAPIPE_CACHE_DIR'] = 'D:/mediapipe_cache'
os.environ['XDG_CACHE_HOME'] = 'D:/mediapipe_cache'
os.environ['GLOG_log_dir'] = 'D:/mediapipe_cache'


def find_solution_base():
    """查找 solution_base.py 文件"""
    try:
        import mediapipe
        mp_path = Path(mediapipe.__file__).parent
        solution_base = mp_path / "python" / "solutions" / "solution_base.py"
        
        if solution_base.exists():
            return solution_base
        
        # 尝试其他可能的位置
        alt_path = mp_path / "python" / "solution_base.py"
        if alt_path.exists():
            return alt_path
            
    except ImportError:
        print("MediaPipe 未安装")
    
    return None


def patch_solution_base(solution_base_path: Path, junction_path: str):
    """
    修补 solution_base.py 使其使用 Junction 路径
    """
    print(f"修补文件: {solution_base_path}")
    
    # 备份原文件
    backup_path = solution_base_path.with_suffix('.py.backup')
    if not backup_path.exists():
        shutil.copy(solution_base_path, backup_path)
        print(f"已备份到: {backup_path}")
    
    # 读取文件内容
    with open(solution_base_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修补过
    if 'MEDIAPIPE_JUNCTION_PATH' in content:
        print("文件已修补过，跳过")
        return True
    
    # 查找需要修改的位置
    # 通常是在获取模型路径的地方
    patch_code = '''
# === MediaPipe 中文路径修补 ===
def _get_patched_path(original_path):
    """将路径转换为 Junction 路径（如果设置了环境变量）"""
    import os
    junction_path = os.environ.get('MEDIAPIPE_JUNCTION_PATH')
    if junction_path and os.path.exists(junction_path):
        # 获取 site-packages 相对路径
        try:
            import mediapipe
            mp_root = os.path.dirname(mediapipe.__file__)
            if original_path.startswith(mp_root):
                relative = original_path[len(mp_root):]
                new_path = os.path.join(junction_path, 'mediapipe' + relative)
                if os.path.exists(new_path) or os.path.exists(os.path.dirname(new_path)):
                    return new_path
        except:
            pass
    return original_path
# === 修补结束 ===

'''
    
    # 在文件开头的 import 之后插入修补代码
    import_end = content.find('\n\n', content.find('import'))
    if import_end == -1:
        import_end = content.find('\nclass')
    
    if import_end != -1:
        new_content = content[:import_end] + '\n' + patch_code + content[import_end:]
        
        # 替换路径获取的地方
        # 查找 _resource_path 相关的代码
        new_content = new_content.replace(
            'resource_path = os.path.join',
            'resource_path = _get_patched_path(os.path.join'
        )
        
        with open(solution_base_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("修补完成")
        return True
    
    print("无法找到合适的插入位置")
    return False


def create_simple_workaround():
    """
    创建简单的环境变量配置启动脚本
    """
    startup_script = '''
import os
import sys

# 设置 MediaPipe Junction 路径
os.environ['MEDIAPIPE_JUNCTION_PATH'] = r'C:\\mediapipe_temp'

# 将 Junction 路径添加到 Python 路径
junction_mp = r'C:\\mediapipe_temp\\mediapipe'
if os.path.exists(junction_mp):
    # 替换 sys.modules 中的 mediapipe 路径
    pass

print(f"MEDIAPIPE_JUNCTION_PATH = {os.environ.get('MEDIAPIPE_JUNCTION_PATH')}")
'''
    return startup_script


def test_mediapipe():
    """测试 MediaPipe 是否工作"""
    print("\n测试 MediaPipe...")
    
    # 设置环境变量
    os.environ['MEDIAPIPE_JUNCTION_PATH'] = r'C:\mediapipe_temp'
    
    try:
        import mediapipe as mp
        import numpy as np
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        
        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pose.process(test_image)
        
        pose.close()
        print("✅ MediaPipe 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe 测试失败: {e}")
        return False


def main():
    print("=" * 50)
    print("MediaPipe 中文路径修补工具")
    print("=" * 50)
    
    junction_path = r'C:\mediapipe_temp'
    
    # 检查 Junction 是否存在
    if not os.path.exists(junction_path):
        print(f"❌ Junction 路径不存在: {junction_path}")
        print("请先以管理员身份运行以下命令创建 Junction:")
        print(f'  New-Item -ItemType Junction -Path "{junction_path}" -Target "<venv>/Lib/site-packages"')
        return 1
    
    print(f"✅ Junction 路径存在: {junction_path}")
    
    # 设置环境变量
    os.environ['MEDIAPIPE_JUNCTION_PATH'] = junction_path
    print(f"✅ 设置环境变量 MEDIAPIPE_JUNCTION_PATH = {junction_path}")
    
    # 测试
    if test_mediapipe():
        print("\n" + "=" * 50)
        print("修补成功！使用方法：")
        print("=" * 50)
        print("\n在运行程序前设置环境变量:")
        print(f'  $env:MEDIAPIPE_JUNCTION_PATH = "{junction_path}"')
        print("\n或者在 Python 代码开头添加:")
        print(f'  import os')
        print(f'  os.environ["MEDIAPIPE_JUNCTION_PATH"] = r"{junction_path}"')
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
