"""
手部3D骨骼模型
表示单只手的21个关键点及骨骼连接
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .coordinate_transformer import Point3D, calculate_angle, calculate_distance
from .constants import (
    HandLandmark,
    HAND_LANDMARK_NAMES,
    HAND_CONNECTIONS,
    FINGER_LANDMARKS,
    HAND_MEASUREMENT_BONES,
    HAND_JOINT_ANGLES,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HandSkeleton3D:
    """
    3D手部骨骼模型
    
    表示单只手的21个关键点及骨骼连接
    
    Example:
        >>> skeleton = HandSkeleton3D.from_points(points_3d, "left")
        >>> palm_length = skeleton.get_palm_length()
        >>> finger_angle = skeleton.get_finger_angle("index", "pip")
    """
    
    def __init__(self, joints: Dict[str, Point3D], handedness: str = "unknown"):
        """
        初始化手部骨骼模型
        
        Args:
            joints: 关节点字典，键为关节名称，值为3D坐标
            handedness: 左手或右手 ("left" / "right")
        """
        self.joints = joints
        self.handedness = handedness
    
    @classmethod
    def from_points(cls, points: List[Point3D], handedness: str = "unknown") -> 'HandSkeleton3D':
        """
        从3D点列表创建骨骼模型
        
        Args:
            points: 21个3D关键点列表（按MediaPipe Hand顺序）
            handedness: 左手或右手
        
        Returns:
            HandSkeleton3D 实例
        """
        if len(points) != 21:
            logger.warning(f"手部关键点数量不是21: {len(points)}")
        
        joints = {}
        for i, point in enumerate(points):
            name = HAND_LANDMARK_NAMES.get(i)
            if name:
                joints[name] = point
        
        return cls(joints, handedness)

    
    def get_joint(self, name: str) -> Optional[Point3D]:
        """获取指定关节的3D坐标"""
        return self.joints.get(name)
    
    def get_palm_length(self) -> float:
        """
        获取手掌长度
        
        计算: 手腕中心 (landmark 0) 到中指MCP (landmark 9) 的距离
        
        Returns:
            手掌长度（米）
        """
        wrist = self.joints.get('wrist')
        middle_mcp = self.joints.get('middle_finger_mcp')
        
        if wrist and middle_mcp and wrist.is_valid() and middle_mcp.is_valid():
            return wrist.distance_to(middle_mcp)
        return 0.0
    
    def get_hand_width(self) -> float:
        """
        获取手宽
        
        计算: 拇指CMC (landmark 1) 到小指MCP (landmark 17) 的距离
        
        Returns:
            手宽（米）
        """
        thumb_cmc = self.joints.get('thumb_cmc')
        pinky_mcp = self.joints.get('pinky_mcp')
        
        if thumb_cmc and pinky_mcp and thumb_cmc.is_valid() and pinky_mcp.is_valid():
            return thumb_cmc.distance_to(pinky_mcp)
        return 0.0
    
    def get_thumb_span(self) -> float:
        """
        获取拇指跨度
        
        计算: 拇指Tip (landmark 4) 到食指Tip (landmark 8) 的距离
        
        Returns:
            拇指跨度（米）
        """
        thumb_tip = self.joints.get('thumb_tip')
        index_tip = self.joints.get('index_finger_tip')
        
        if thumb_tip and index_tip and thumb_tip.is_valid() and index_tip.is_valid():
            return thumb_tip.distance_to(index_tip)
        return 0.0
    
    def get_finger_length(self, finger: str) -> float:
        """
        获取手指长度
        
        计算: MCP到Tip的累加距离
        
        Args:
            finger: 手指名称 ("thumb", "index", "middle", "ring", "pinky")
        
        Returns:
            手指长度（米）
        """
        if finger not in FINGER_LANDMARKS:
            return 0.0
        
        landmarks = FINGER_LANDMARKS[finger]
        total_length = 0.0
        
        for i in range(len(landmarks) - 1):
            start_name = HAND_LANDMARK_NAMES.get(landmarks[i])
            end_name = HAND_LANDMARK_NAMES.get(landmarks[i + 1])
            
            if start_name and end_name:
                start = self.joints.get(start_name)
                end = self.joints.get(end_name)
                
                if start and end and start.is_valid() and end.is_valid():
                    total_length += start.distance_to(end)
        
        return total_length
    
    def get_all_finger_lengths(self) -> Dict[str, float]:
        """获取所有手指长度"""
        return {
            finger: self.get_finger_length(finger)
            for finger in FINGER_LANDMARKS.keys()
        }
    
    def get_finger_angle(self, finger: str, joint: str) -> float:
        """
        获取手指关节角度
        
        Args:
            finger: 手指名称 ("thumb", "index", "middle", "ring", "pinky")
            joint: 关节名称 ("mcp", "pip", "dip" 或 "ip" for thumb)
        
        Returns:
            角度（度）
        """
        angle_key = f"{finger}_{joint}"
        
        if angle_key not in HAND_JOINT_ANGLES:
            return 0.0
        
        idx1, idx2, idx3 = HAND_JOINT_ANGLES[angle_key]
        name1 = HAND_LANDMARK_NAMES.get(idx1)
        name2 = HAND_LANDMARK_NAMES.get(idx2)
        name3 = HAND_LANDMARK_NAMES.get(idx3)
        
        p1 = self.joints.get(name1)
        p2 = self.joints.get(name2)
        p3 = self.joints.get(name3)
        
        if p1 and p2 and p3 and p1.is_valid() and p2.is_valid() and p3.is_valid():
            return calculate_angle(p1, p2, p3)
        return 0.0
    
    def get_all_joint_angles(self) -> Dict[str, float]:
        """获取所有关节角度"""
        angles = {}
        for angle_name in HAND_JOINT_ANGLES.keys():
            parts = angle_name.split('_')
            finger = parts[0]
            joint = parts[1]
            angles[angle_name] = self.get_finger_angle(finger, joint)
        return angles
    
    def get_palm_plane(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取手掌平面
        
        使用手腕、食指MCP、小指MCP三点定义平面
        
        Returns:
            (法向量, 中心点) 元组
        """
        wrist = self.joints.get('wrist')
        index_mcp = self.joints.get('index_finger_mcp')
        pinky_mcp = self.joints.get('pinky_mcp')
        
        if not all([wrist, index_mcp, pinky_mcp]):
            return np.zeros(3), np.zeros(3)
        
        if not all([wrist.is_valid(), index_mcp.is_valid(), pinky_mcp.is_valid()]):
            return np.zeros(3), np.zeros(3)
        
        # 计算两个向量
        v1 = np.array([
            index_mcp.x - wrist.x,
            index_mcp.y - wrist.y,
            index_mcp.z - wrist.z
        ])
        v2 = np.array([
            pinky_mcp.x - wrist.x,
            pinky_mcp.y - wrist.y,
            pinky_mcp.z - wrist.z
        ])
        
        # 法向量 = v1 × v2
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        # 中心点
        center = np.array([
            (wrist.x + index_mcp.x + pinky_mcp.x) / 3,
            (wrist.y + index_mcp.y + pinky_mcp.y) / 3,
            (wrist.z + index_mcp.z + pinky_mcp.z) / 3
        ])
        
        return normal, center
    
    def get_tip_to_mcp_distance(self, finger: str) -> float:
        """
        获取指尖到MCP的直线距离
        
        用于手势识别（判断手指是否弯曲）
        
        Args:
            finger: 手指名称
        
        Returns:
            距离（米）
        """
        if finger not in FINGER_LANDMARKS:
            return 0.0
        
        landmarks = FINGER_LANDMARKS[finger]
        mcp_idx = landmarks[0]
        tip_idx = landmarks[-1]
        
        mcp_name = HAND_LANDMARK_NAMES.get(mcp_idx)
        tip_name = HAND_LANDMARK_NAMES.get(tip_idx)
        
        mcp = self.joints.get(mcp_name)
        tip = self.joints.get(tip_name)
        
        if mcp and tip and mcp.is_valid() and tip.is_valid():
            return mcp.distance_to(tip)
        return 0.0
    
    def get_overall_confidence(self) -> float:
        """获取整体置信度"""
        if not self.joints:
            return 0.0
        
        # 使用所有点的置信度
        all_confidences = [j.confidence for j in self.joints.values()]
        if all_confidences:
            avg_conf = sum(all_confidences) / len(all_confidences)
            # 如果平均置信度为0但有有效的3D点，返回一个默认置信度
            if avg_conf == 0:
                valid_count = sum(1 for j in self.joints.values() if j.is_valid())
                if valid_count > 10:  # 超过一半的点有效
                    return 0.8  # 返回默认高置信度
            return avg_conf
        
        return 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'handedness': self.handedness,
            'joints': {
                name: {
                    'x': p.x, 'y': p.y, 'z': p.z,
                    'confidence': p.confidence
                }
                for name, p in self.joints.items()
            },
            'measurements': {
                'palm_length': self.get_palm_length(),
                'hand_width': self.get_hand_width(),
                'thumb_span': self.get_thumb_span(),
                'finger_lengths': self.get_all_finger_lengths()
            }
        }
    
    def to_array(self) -> np.ndarray:
        """
        转换为numpy数组
        
        Returns:
            shape (21, 4) 的数组，每行为 [x, y, z, confidence]
        """
        arr = np.zeros((21, 4))
        
        for idx, name in HAND_LANDMARK_NAMES.items():
            if name in self.joints:
                joint = self.joints[name]
                arr[idx] = [joint.x, joint.y, joint.z, joint.confidence]
        
        return arr
    
    @property
    def connections(self) -> List[Tuple[str, str]]:
        """骨骼连接列表（用于绘制）"""
        connections = []
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_name = HAND_LANDMARK_NAMES.get(start_idx)
            end_name = HAND_LANDMARK_NAMES.get(end_idx)
            if start_name and end_name:
                connections.append((start_name, end_name))
        return connections
