"""
线性尺寸测量
计算身高、肩宽、臂长、腿长等线性尺寸
"""
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.skeleton import Skeleton3D
from ..core.coordinate_transformer import Point3D, calculate_distance
from ..core.constants import PoseLandmark, LANDMARK_NAMES
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LinearMeasurements:
    """
    线性尺寸测量类
    """
    
    @staticmethod
    def calculate_height(skeleton: Skeleton3D) -> float:
        """
        估算身高

        使用Y轴垂直距离累加各骨骼段：
        头顶补偿 + 颈 + 躯干 + 大腿 + 小腿 + 脚底补偿

        Y轴垂直距离比3D欧几里得距离更稳定准确：
        - 身高是垂直量，Y轴距离直接对应
        - 不受身体前倾/侧弯影响
        - 帧间抖动更小

        补偿值为固定值（基于经验校准）：
        - 头顶补偿(鼻梁→头顶) = 0.21m
        - 脚底补偿(踝→脚底) = 0.09m
        """
        nose = skeleton.get_joint('nose')
        left_shoulder = skeleton.get_joint('left_shoulder')
        right_shoulder = skeleton.get_joint('right_shoulder')
        left_hip = skeleton.get_joint('left_hip')
        right_hip = skeleton.get_joint('right_hip')
        left_knee = skeleton.get_joint('left_knee')
        right_knee = skeleton.get_joint('right_knee')
        left_ankle = skeleton.get_joint('left_ankle')
        right_ankle = skeleton.get_joint('right_ankle')

        if not nose:
            return 0.0

        shoulder = left_shoulder or right_shoulder
        if not shoulder:
            return 0.0
        if left_shoulder and right_shoulder:
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        else:
            shoulder_center_y = shoulder.y

        hip = left_hip or right_hip
        if not hip:
            return 0.0
        if left_hip and right_hip:
            hip_center_y = (left_hip.y + right_hip.y) / 2
        else:
            hip_center_y = hip.y

        knee = left_knee or right_knee
        ankle = left_ankle or right_ankle
        if not knee or not ankle:
            return 0.0

        neck_vert = abs(nose.y - shoulder_center_y)
        torso_vert = abs(shoulder_center_y - hip_center_y)

        if left_knee and left_hip:
            left_thigh_vert = abs(left_hip.y - left_knee.y)
        else:
            left_thigh_vert = None
        if right_knee and right_hip:
            right_thigh_vert = abs(right_hip.y - right_knee.y)
        else:
            right_thigh_vert = None
        if left_thigh_vert is not None and right_thigh_vert is not None:
            thigh_vert = (left_thigh_vert + right_thigh_vert) / 2
        elif left_thigh_vert is not None:
            thigh_vert = left_thigh_vert
        elif right_thigh_vert is not None:
            thigh_vert = right_thigh_vert
        else:
            return 0.0

        if left_knee and left_ankle:
            left_calf_vert = abs(left_knee.y - left_ankle.y)
        else:
            left_calf_vert = None
        if right_knee and right_ankle:
            right_calf_vert = abs(right_knee.y - right_ankle.y)
        else:
            right_calf_vert = None
        if left_calf_vert is not None and right_calf_vert is not None:
            calf_vert = (left_calf_vert + right_calf_vert) / 2
        elif left_calf_vert is not None:
            calf_vert = left_calf_vert
        elif right_calf_vert is not None:
            calf_vert = right_calf_vert
        else:
            return 0.0

        # 固定补偿值（经验校准）
        head_top = 0.21     # 鼻梁→头顶
        foot_bottom = 0.09   # 踝→脚底

        base_height = head_top + neck_vert + torso_vert + thigh_vert + calf_vert + foot_bottom

        logger.debug(f"身高组成(Y轴): 头顶={head_top:.3f}, "
                    f"颈={neck_vert:.3f}, 躯干={torso_vert:.3f}, "
                    f"大腿={thigh_vert:.3f}, 小腿={calf_vert:.3f}, "
                    f"脚底={foot_bottom:.3f}, "
                    f"合计={base_height:.3f}m")

        return base_height
    
    @staticmethod
    def calculate_shoulder_width(skeleton: Skeleton3D) -> float:
        """计算肩宽"""
        left_shoulder = skeleton.get_joint('left_shoulder')
        right_shoulder = skeleton.get_joint('right_shoulder')
        
        if not left_shoulder or not right_shoulder:
            return 0.0
        
        return calculate_distance(left_shoulder, right_shoulder)
    
    @staticmethod
    def calculate_arm_span(skeleton: Skeleton3D) -> float:
        """计算臂展"""
        left_wrist = skeleton.get_joint('left_wrist')
        right_wrist = skeleton.get_joint('right_wrist')
        
        if not left_wrist or not right_wrist:
            return 0.0
        
        return calculate_distance(left_wrist, right_wrist)
    
    @staticmethod
    def calculate_leg_length(skeleton: Skeleton3D) -> float:
        """计算腿长"""
        left_hip = skeleton.get_joint('left_hip')
        right_hip = skeleton.get_joint('right_hip')
        left_ankle = skeleton.get_joint('left_ankle')
        right_ankle = skeleton.get_joint('right_ankle')
        
        if not all([left_hip, right_hip, left_ankle, right_ankle]):
            return 0.0
        
        left_leg = calculate_distance(left_hip, left_ankle)
        right_leg = calculate_distance(right_hip, right_ankle)
        
        return (left_leg + right_leg) / 2
    
    @staticmethod
    def calculate_sitting_height(skeleton: Skeleton3D) -> float:
        """计算坐高"""
        nose = skeleton.get_joint('nose')
        left_hip = skeleton.get_joint('left_hip')
        right_hip = skeleton.get_joint('right_hip')

        if not nose or not left_hip or not right_hip:
            return 0.0

        hip_center_y = (left_hip.y + right_hip.y) / 2

        return abs(nose.y - hip_center_y)

    @staticmethod
    def calculate_pelvic_width(skeleton: Skeleton3D) -> float:
        """计算骨盆宽（髂嵴间宽）"""
        left_hip = skeleton.get_joint('left_hip')
        right_hip = skeleton.get_joint('right_hip')
        if not left_hip or not right_hip:
            return 0.0
        return calculate_distance(left_hip, right_hip)

    @staticmethod
    def calculate_upper_limb_length(skeleton: Skeleton3D) -> float:
        """计算上肢长（肩峰→桡骨茎突，取左右平均）"""
        left_shoulder = skeleton.get_joint('left_shoulder')
        left_wrist = skeleton.get_joint('left_wrist')
        right_shoulder = skeleton.get_joint('right_shoulder')
        right_wrist = skeleton.get_joint('right_wrist')

        left_len = calculate_distance(left_shoulder, left_wrist) if left_shoulder and left_wrist else 0.0
        right_len = calculate_distance(right_shoulder, right_wrist) if right_shoulder and right_wrist else 0.0

        if left_len > 0 and right_len > 0:
            return (left_len + right_len) / 2
        return left_len or right_len

    @staticmethod
    def calculate_lower_limb_length(skeleton: Skeleton3D) -> float:
        """计算下肢长（髂前上棘→内踝，取左右平均）"""
        left_hip = skeleton.get_joint('left_hip')
        left_ankle = skeleton.get_joint('left_ankle')
        right_hip = skeleton.get_joint('right_hip')
        right_ankle = skeleton.get_joint('right_ankle')

        left_len = calculate_distance(left_hip, left_ankle) if left_hip and left_ankle else 0.0
        right_len = calculate_distance(right_hip, right_ankle) if right_hip and right_ankle else 0.0

        if left_len > 0 and right_len > 0:
            return (left_len + right_len) / 2
        return left_len or right_len

    @staticmethod
    def calculate_trunk_length(skeleton: Skeleton3D) -> float:
        """计算颈-臀长（躯干长，nose→hip_center 垂直投影）"""
        nose = skeleton.get_joint('nose')
        left_hip = skeleton.get_joint('left_hip')
        right_hip = skeleton.get_joint('right_hip')

        if not nose or not left_hip or not right_hip:
            return 0.0

        hip_center_y = (left_hip.y + right_hip.y) / 2
        return abs(nose.y - hip_center_y)

    @staticmethod
    def calculate_foot_length(skeleton: Skeleton3D) -> float:
        """计算足长（踝→脚趾，取左右平均）"""
        left_ankle = skeleton.get_joint('left_ankle')
        left_foot = skeleton.get_joint('left_foot_index')
        right_ankle = skeleton.get_joint('right_ankle')
        right_foot = skeleton.get_joint('right_foot_index')

        left_len = calculate_distance(left_ankle, left_foot) if left_ankle and left_foot else 0.0
        right_len = calculate_distance(right_ankle, right_foot) if right_ankle and right_foot else 0.0

        if left_len > 0 and right_len > 0:
            return (left_len + right_len) / 2
        return left_len or right_len
    
    @staticmethod
    def calculate_bike_fitting_bones(skeleton: Skeleton3D) -> Dict[str, float]:
        bones = {}

        left_shoulder = skeleton.get_joint('left_shoulder')
        right_shoulder = skeleton.get_joint('right_shoulder')
        left_elbow = skeleton.get_joint('left_elbow')
        right_elbow = skeleton.get_joint('right_elbow')
        left_wrist = skeleton.get_joint('left_wrist')
        right_wrist = skeleton.get_joint('right_wrist')
        left_hip = skeleton.get_joint('left_hip')
        right_hip = skeleton.get_joint('right_hip')
        left_knee = skeleton.get_joint('left_knee')
        right_knee = skeleton.get_joint('right_knee')
        left_ankle = skeleton.get_joint('left_ankle')
        right_ankle = skeleton.get_joint('right_ankle')
        left_foot = skeleton.get_joint('left_foot_index')
        right_foot = skeleton.get_joint('right_foot_index')
        left_hand = skeleton.get_joint('left_index')
        right_hand = skeleton.get_joint('right_index')
        nose = skeleton.get_joint('nose')

        # 虚拟中心点
        if left_shoulder and right_shoulder:
            shoulder_center = Point3D(
                x=(left_shoulder.x + right_shoulder.x) / 2,
                y=(left_shoulder.y + right_shoulder.y) / 2,
                z=(left_shoulder.z + right_shoulder.z) / 2
            )
        else:
            shoulder_center = None

        if left_hip and right_hip:
            hip_center = Point3D(
                x=(left_hip.x + right_hip.x) / 2,
                y=(left_hip.y + right_hip.y) / 2,
                z=(left_hip.z + right_hip.z) / 2
            )
        else:
            hip_center = None

        if shoulder_center and hip_center:
            spine_mid = Point3D(
                x=(shoulder_center.x + hip_center.x) / 2,
                y=(shoulder_center.y + hip_center.y) / 2,
                z=(shoulder_center.z + hip_center.z) / 2
            )
        else:
            spine_mid = None

        # 脊柱分段
        if hip_center and spine_mid:
            bones['spine_base_to_spine_mid'] = calculate_distance(hip_center, spine_mid)
        if spine_mid and shoulder_center:
            bones['spine_mid_to_spine_shoulder'] = calculate_distance(spine_mid, shoulder_center)
        if shoulder_center and nose:
            bones['spine_shoulder_to_head'] = calculate_distance(shoulder_center, nose)

        # 肩部到两侧肩关节
        if shoulder_center and left_shoulder:
            bones['spine_shoulder_to_shoulder_left'] = calculate_distance(shoulder_center, left_shoulder)
        if shoulder_center and right_shoulder:
            bones['spine_shoulder_to_shoulder_right'] = calculate_distance(shoulder_center, right_shoulder)

        # 躯干长度（保留兼容旧接口）
        if shoulder_center and hip_center:
            bones['躯干长度'] = calculate_distance(shoulder_center, hip_center)

        # 左臂分段
        if left_shoulder and left_elbow:
            bones['left_shoulder_to_left_elbow'] = calculate_distance(left_shoulder, left_elbow)
            bones['左大臂长度'] = bones['left_shoulder_to_left_elbow']
        if left_elbow and left_wrist:
            bones['left_elbow_to_left_wrist'] = calculate_distance(left_elbow, left_wrist)
            bones['左前臂长度'] = bones['left_elbow_to_left_wrist']
        if left_wrist and left_hand:
            bones['left_wrist_to_left_hand'] = calculate_distance(left_wrist, left_hand)

        # 右臂分段
        if right_shoulder and right_elbow:
            bones['right_shoulder_to_right_elbow'] = calculate_distance(right_shoulder, right_elbow)
            bones['右大臂长度'] = bones['right_shoulder_to_right_elbow']
        if right_elbow and right_wrist:
            bones['right_elbow_to_right_wrist'] = calculate_distance(right_elbow, right_wrist)
            bones['右前臂长度'] = bones['right_elbow_to_right_wrist']
        if right_wrist and right_hand:
            bones['right_wrist_to_right_hand'] = calculate_distance(right_wrist, right_hand)

        # 髋部分段（脊柱底部到两侧髋关节）
        if hip_center and left_hip:
            bones['spine_base_to_left_hip'] = calculate_distance(hip_center, left_hip)
        if hip_center and right_hip:
            bones['spine_base_to_right_hip'] = calculate_distance(hip_center, right_hip)

        # 左腿分段
        if left_hip and left_knee:
            bones['left_hip_to_left_knee'] = calculate_distance(left_hip, left_knee)
            bones['左大腿长度'] = bones['left_hip_to_left_knee']
        if left_knee and left_ankle:
            bones['left_knee_to_left_ankle'] = calculate_distance(left_knee, left_ankle)
            bones['左小腿长度'] = bones['left_knee_to_left_ankle']
        if left_ankle and left_foot:
            bones['left_ankle_to_left_foot'] = calculate_distance(left_ankle, left_foot)

        # 右腿分段
        if right_hip and right_knee:
            bones['right_hip_to_right_knee'] = calculate_distance(right_hip, right_knee)
            bones['右大腿长度'] = bones['right_hip_to_right_knee']
        if right_knee and right_ankle:
            bones['right_knee_to_right_ankle'] = calculate_distance(right_knee, right_ankle)
            bones['右小腿长度'] = bones['right_knee_to_right_ankle']
        if right_ankle and right_foot:
            bones['right_ankle_to_right_foot'] = calculate_distance(right_ankle, right_foot)

        return bones
