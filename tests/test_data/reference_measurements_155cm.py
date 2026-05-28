"""
参考测量数据 - 155cm 身高
用于测试和校准骨骼测量算法
"""

# 155cm 身高的参考测量数据（单位：mm）
REFERENCE_DATA_155CM = {
    # 基本信息
    "height": 1550.7,  # 身高（mm）
    
    # 脊柱段
    "spine_base_to_mid": 276.9,          # 脊柱底部到中部
    "spine_mid_to_shoulder": 276.9,      # 脊柱中部到肩部
    "spine_shoulder_to_head": 210.4,     # 脊柱肩部到头部
    
    # 左臂
    "spine_shoulder_to_left_shoulder": 169.9,   # 脊柱肩部到左肩
    "left_shoulder_to_left_elbow": 264.9,       # 左肩到左肘
    "left_elbow_to_left_wrist": 230.0,          # 左肘到左腕
    "left_wrist_to_left_hand": 69.8,            # 左腕到左手
    
    # 右臂
    "spine_shoulder_to_right_shoulder": 169.9,  # 脊柱肩部到右肩
    "right_shoulder_to_right_elbow": 280.5,     # 右肩到右肘
    "right_elbow_to_right_wrist": 226.5,        # 右肘到右腕
    "right_wrist_to_right_hand": 70.7,          # 右腕到右手
    
    # 左腿
    "spine_base_to_left_hip": 97.5,      # 脊柱底部到左髋
    "left_hip_to_left_knee": 378.0,      # 左髋到左膝
    "left_knee_to_left_ankle": 303.6,    # 左膝到左踝
    "left_ankle_to_left_foot": 130.2,    # 左踝到左脚
    
    # 右腿
    "spine_base_to_right_hip": 97.5,     # 脊柱底部到右髋
    "right_hip_to_right_knee": 357.7,    # 右髋到右膝
    "right_knee_to_right_ankle": 303.3,  # 右膝到右踝
    "right_ankle_to_right_foot": 145.3,  # 右踝到右脚
}

# 转换为米（用于计算）
REFERENCE_DATA_155CM_METERS = {
    key: value / 1000.0 for key, value in REFERENCE_DATA_155CM.items()
}

# 计算派生数据
DERIVED_DATA_155CM = {
    # 总脊柱长度
    "total_spine": REFERENCE_DATA_155CM["spine_base_to_mid"] + 
                   REFERENCE_DATA_155CM["spine_mid_to_shoulder"] + 
                   REFERENCE_DATA_155CM["spine_shoulder_to_head"],  # 764.2mm
    
    # 肩宽（左右肩之间）
    "shoulder_width": REFERENCE_DATA_155CM["spine_shoulder_to_left_shoulder"] + 
                      REFERENCE_DATA_155CM["spine_shoulder_to_right_shoulder"],  # 339.8mm
    
    # 髋宽（左右髋之间）
    "hip_width": REFERENCE_DATA_155CM["spine_base_to_left_hip"] + 
                 REFERENCE_DATA_155CM["spine_base_to_right_hip"],  # 195.0mm
    
    # 左臂总长
    "left_arm_total": REFERENCE_DATA_155CM["left_shoulder_to_left_elbow"] + 
                      REFERENCE_DATA_155CM["left_elbow_to_left_wrist"] + 
                      REFERENCE_DATA_155CM["left_wrist_to_left_hand"],  # 564.7mm
    
    # 右臂总长
    "right_arm_total": REFERENCE_DATA_155CM["right_shoulder_to_right_elbow"] + 
                       REFERENCE_DATA_155CM["right_elbow_to_right_wrist"] + 
                       REFERENCE_DATA_155CM["right_wrist_to_right_hand"],  # 577.7mm
    
    # 左腿总长
    "left_leg_total": REFERENCE_DATA_155CM["left_hip_to_left_knee"] + 
                      REFERENCE_DATA_155CM["left_knee_to_left_ankle"] + 
                      REFERENCE_DATA_155CM["left_ankle_to_left_foot"],  # 811.8mm
    
    # 右腿总长
    "right_leg_total": REFERENCE_DATA_155CM["right_hip_to_right_knee"] + 
                       REFERENCE_DATA_155CM["right_knee_to_right_ankle"] + 
                       REFERENCE_DATA_155CM["right_ankle_to_right_foot"],  # 806.3mm
}

# 打印摘要
def print_summary():
    """打印参考数据摘要"""
    print("=" * 60)
    print("参考测量数据 - 155cm 身高")
    print("=" * 60)
    print(f"\n身高: {REFERENCE_DATA_155CM['height']:.1f}mm ({REFERENCE_DATA_155CM['height']/10:.1f}cm)")
    print(f"\n脊柱:")
    print(f"  底部到中部: {REFERENCE_DATA_155CM['spine_base_to_mid']:.1f}mm")
    print(f"  中部到肩部: {REFERENCE_DATA_155CM['spine_mid_to_shoulder']:.1f}mm")
    print(f"  肩部到头部: {REFERENCE_DATA_155CM['spine_shoulder_to_head']:.1f}mm")
    print(f"  总长: {DERIVED_DATA_155CM['total_spine']:.1f}mm")
    
    print(f"\n左臂:")
    print(f"  肩到肘: {REFERENCE_DATA_155CM['left_shoulder_to_left_elbow']:.1f}mm")
    print(f"  肘到腕: {REFERENCE_DATA_155CM['left_elbow_to_left_wrist']:.1f}mm")
    print(f"  腕到手: {REFERENCE_DATA_155CM['left_wrist_to_left_hand']:.1f}mm")
    print(f"  总长: {DERIVED_DATA_155CM['left_arm_total']:.1f}mm")
    
    print(f"\n右臂:")
    print(f"  肩到肘: {REFERENCE_DATA_155CM['right_shoulder_to_right_elbow']:.1f}mm")
    print(f"  肘到腕: {REFERENCE_DATA_155CM['right_elbow_to_right_wrist']:.1f}mm")
    print(f"  腕到手: {REFERENCE_DATA_155CM['right_wrist_to_right_hand']:.1f}mm")
    print(f"  总长: {DERIVED_DATA_155CM['right_arm_total']:.1f}mm")
    
    print(f"\n左腿:")
    print(f"  髋到膝: {REFERENCE_DATA_155CM['left_hip_to_left_knee']:.1f}mm")
    print(f"  膝到踝: {REFERENCE_DATA_155CM['left_knee_to_left_ankle']:.1f}mm")
    print(f"  踝到脚: {REFERENCE_DATA_155CM['left_ankle_to_left_foot']:.1f}mm")
    print(f"  总长: {DERIVED_DATA_155CM['left_leg_total']:.1f}mm")
    
    print(f"\n右腿:")
    print(f"  髋到膝: {REFERENCE_DATA_155CM['right_hip_to_right_knee']:.1f}mm")
    print(f"  膝到踝: {REFERENCE_DATA_155CM['right_knee_to_right_ankle']:.1f}mm")
    print(f"  踝到脚: {REFERENCE_DATA_155CM['right_ankle_to_right_foot']:.1f}mm")
    print(f"  总长: {DERIVED_DATA_155CM['right_leg_total']:.1f}mm")
    
    print(f"\n其他:")
    print(f"  肩宽: {DERIVED_DATA_155CM['shoulder_width']:.1f}mm")
    print(f"  髋宽: {DERIVED_DATA_155CM['hip_width']:.1f}mm")
    print("=" * 60)


if __name__ == "__main__":
    print_summary()
