"""
将现有的测量 JSON 文件转换为易读的 CSV 表格

用法: python scripts/convert_json_to_csv.py [json_file]

如果不指定文件，会转换 data/sessions 目录下所有的测量 JSON 文件
"""
import csv
import json
import sys
from pathlib import Path

# 字段名称中英文映射
FIELD_NAMES_CN = {
    # 基本信息
    'height': '身高',
    'shoulder_width': '肩宽',
    'hip_width': '髋宽',
    'arm_span': '臂展',
    'torso_length': '躯干长',
    'neck_length': '颈长',
    
    # 手臂
    'left_arm_length': '左臂长',
    'right_arm_length': '右臂长',
    'left_upper_arm': '左上臂长',
    'right_upper_arm': '右上臂长',
    'left_forearm': '左前臂长',
    'right_forearm': '右前臂长',
    
    # 腿部
    'left_leg_length': '左腿长',
    'right_leg_length': '右腿长',
    'left_thigh': '左大腿长',
    'right_thigh': '右大腿长',
    'left_calf': '左小腿长',
    'right_calf': '右小腿长',
    
    # Bike Fitting 躯干与脊柱
    'spine_base_to_spine_mid': '脊柱底部到中部',
    'spine_mid_to_spine_shoulder': '脊柱中部到肩部',
    'spine_shoulder_to_head': '脊柱肩部到头部',
    
    # Bike Fitting 上肢（左侧）
    'spine_shoulder_to_shoulder_left': '脊柱肩部到左肩',
    'shoulder_left_to_elbow_left': '左肩到左肘',
    'elbow_left_to_wrist_left': '左肘到左腕',
    'wrist_left_to_hand_left': '左腕到左手',
    
    # Bike Fitting 上肢（右侧）
    'spine_shoulder_to_shoulder_right': '脊柱肩部到右肩',
    'shoulder_right_to_elbow_right': '右肩到右肘',
    'elbow_right_to_wrist_right': '右肘到右腕',
    'wrist_right_to_hand_right': '右腕到右手',
    
    # Bike Fitting 下肢（左侧）
    'spine_base_to_hip_left': '脊柱底部到左髋',
    'hip_left_to_knee_left': '左髋到左膝',
    'knee_left_to_ankle_left': '左膝到左踝',
    'ankle_left_to_foot_left': '左踝到左脚',
    
    # Bike Fitting 下肢（右侧）
    'spine_base_to_hip_right': '脊柱底部到右髋',
    'hip_right_to_knee_right': '右髋到右膝',
    'knee_right_to_ankle_right': '右膝到右踝',
    'ankle_right_to_foot_right': '右踝到右脚',
}

# 需要输出的字段顺序
EXPORT_FIELD_ORDER = [
    'height', 'shoulder_width', 'hip_width', 'arm_span', 'torso_length', 'neck_length',
    'spine_base_to_spine_mid', 'spine_mid_to_spine_shoulder', 'spine_shoulder_to_head',
    'spine_shoulder_to_shoulder_left', 'shoulder_left_to_elbow_left', 
    'elbow_left_to_wrist_left', 'wrist_left_to_hand_left',
    'spine_shoulder_to_shoulder_right', 'shoulder_right_to_elbow_right',
    'elbow_right_to_wrist_right', 'wrist_right_to_hand_right',
    'spine_base_to_hip_left', 'hip_left_to_knee_left',
    'knee_left_to_ankle_left', 'ankle_left_to_foot_left',
    'spine_base_to_hip_right', 'hip_right_to_knee_right',
    'knee_right_to_ankle_right', 'ankle_right_to_foot_right',
]


def convert_json_to_csv(json_path: Path) -> bool:
    """转换单个 JSON 文件为 CSV"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否是测量文件（包含 body 字段）
        if 'body' not in data:
            print(f"  跳过（非测量文件）: {json_path.name}")
            return False
        
        body = data.get('body')
        if not body:
            print(f"  跳过（无身体数据）: {json_path.name}")
            return False
        
        # 生成 CSV 文件
        csv_path = json_path.with_suffix('.csv')
        
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['测量项目', '长度'])
            
            for field in EXPORT_FIELD_ORDER:
                if field in body:
                    value = body[field]
                    # 转换为毫米，保留1位小数，带单位
                    if isinstance(value, (int, float)) and value > 0:
                        mm_value = f"{value * 1000:.1f}mm"
                    else:
                        mm_value = "0.0mm"
                    cn_name = FIELD_NAMES_CN.get(field, field)
                    writer.writerow([cn_name, mm_value])
            
            # 元数据
            writer.writerow([])
            writer.writerow(['--- 测量信息 ---', ''])
            writer.writerow(['测量时间', data.get('timestamp', '')])
            writer.writerow(['质量分数', f"{data.get('quality_score', 0):.1%}"])
            writer.writerow(['融合帧数', data.get('frames_used', 0)])
            writer.writerow(['置信度', f"{data.get('confidence', 0):.1%}"])
            
            if data.get('anatomy_issues'):
                writer.writerow([])
                writer.writerow(['--- 数据质量问题 ---', ''])
                for issue in data['anatomy_issues']:
                    writer.writerow([issue, ''])
        
        print(f"  ✅ 已转换: {csv_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ 转换失败: {json_path.name} - {e}")
        return False


def main():
    if len(sys.argv) > 1:
        # 转换指定文件
        json_path = Path(sys.argv[1])
        if json_path.exists():
            convert_json_to_csv(json_path)
        else:
            print(f"文件不存在: {json_path}")
            return 1
    else:
        # 转换所有测量文件
        sessions_dir = Path('data/sessions')
        if not sessions_dir.exists():
            print(f"目录不存在: {sessions_dir}")
            return 1
        
        print(f"扫描目录: {sessions_dir}")
        json_files = list(sessions_dir.glob('*_full_body_*.json'))
        
        if not json_files:
            print("未找到测量文件")
            return 0
        
        print(f"找到 {len(json_files)} 个测量文件")
        converted = 0
        for json_path in json_files:
            if convert_json_to_csv(json_path):
                converted += 1
        
        print(f"\n完成！成功转换 {converted} 个文件")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
