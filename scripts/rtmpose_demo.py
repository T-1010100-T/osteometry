"""
MediaPipe 演示脚本（兼容入口）

此文件已解耦到 src/ 目录下的模块中。
现在作为兼容入口，直接调用 src/main.py

新的模块结构：
- src/core/camera_controller.py    - 相机控制
- src/core/measurement_collector.py - 采集状态机
- src/core/measurement_engine.py   - 测量引擎
- src/core/data_aggregator.py      - 数据聚合
- src/visualization/ui_renderer.py - UI 渲染
- src/main.py                      - 主入口

推荐使用：
  python src/main.py
  或
  智能数据测试+存储.bat
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    from src.main import main
    sys.exit(main())
