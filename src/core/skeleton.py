"""
3D骨骼模型
表示人体骨骼结构
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    LANDMARK_NAMES,
    LANDMARK_INDICES,
    POSE_CONNECTIONS,
    MEASUREMENT_BONES,
    JOINT_ANGLES,
    PoseLandmark
)
from .coordinate_transformer import Point3D, calculate_angle, calculate_distance
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Bone:
    """骨骼段"""
    name: str                   # 骨骼名称
    start_joint: str            # 起始关节名称
    end_joint: str              # 结束关节名称
    start_point: Point3D        # 起始点3D坐标
    end_point: Point3D          # 结束点3D坐标
    
    @property
    def length(self) -> float:
        """骨骼长度"""
        return self.start_point.distance_to(self.end_point)
    
    @property
    def direction(self) -> np.ndarray:
        """骨骼方向向量（归一化）"""
        v = np.array([
            self.end_point.x - self.start_point.x,
            self.end_point.y - self.start_point.y,
            self.end_point.z - self.start_point.z
        ])
        norm = np.linalg.norm(v)
        if norm == 0:
            return np.zeros(3)
        return v / norm
    
    @property
    def confidence(self) -> float:
        """置信度（取两端点的最小值）"""
        return min(self.start_point.confidence, self.end_point.confidence)


class Skeleton3D:
    """
    3D骨骼模型
    
    表示人体33个关键点及其骨骼连接
    
    Example:
        >>> skeleton = Skeleton3D.from_points(points_3d)
        >>> shoulder_width = skeleton.get_bone_length('shoulder_width')
        >>> elbow_angle = skeleton.get_joint_angle('left_elbow')
    """
    
    def __init__(self, joints: Dict[str, Point3D]):
        """
        初始化骨骼模型
        
        Args:
            joints: 关节点字典，键为关节名称，值为3D坐标
        """
        self.joints = joints
        self._bones: Dict[str, Bone] = {}
        self._build_bones()
    
    def _build_bones(self) -> None:
        """构建骨骼段"""
        for bone_name, (start_idx, end_idx) in MEASUREMENT_BONES.items():
            start_name = LANDMARK_NAMES.get(start_idx)
            end_name = LANDMARK_NAMES.get(end_idx)
            
            if start_name in self.joints and end_name in self.joints:
                self._bones[bone_name] = Bone(
                    name=bone_name,
                    start_joint=start_name,
                    end_joint=end_name,
                    start_point=self.joints[start_name],
                    end_point=self.joints[end_name]
                )
    
    @classmethod
    def from_points(cls, points: List[Point3D]) -> 'Skeleton3D':
        """
        从3D点列表创建骨骼模型
        
        Args:
            points: 33个3D关键点列表（按MediaPipe顺序）
        
        Returns:
            Skeleton3D 实例
        """
        if len(points) != 33:
            logger.warning(f"关键点数量不是33: {len(points)}")
        
        joints = {}
        for i, point in enumerate(points):
            name = LANDMARK_NAMES.get(i)
            if name and point is not None:
                joints[name] = point
        
        return cls(joints)
    
    def get_joint(self, name: str) -> Optional[Point3D]:
        """
        获取指定关节的3D坐标
        
        Args:
            name: 关节名称
        
        Returns:
            Point3D 或 None
        """
        return self.joints.get(name)
    
    def get_joint_by_index(self, index: int) -> Optional[Point3D]:
        """
        通过索引获取关节
        
        Args:
            index: 关节索引 (0-32)
        
        Returns:
            Point3D 或 None
        """
        name = LANDMARK_NAMES.get(index)
        if name:
            return self.joints.get(name)
        return None
    
    def get_bone(self, name: str) -> Optional[Bone]:
        """
        获取指定骨骼段
        
        Args:
            name: 骨骼名称
        
        Returns:
            Bone 或 None
        """
        return self._bones.get(name)
    
    def get_bone_length(self, bone_name: str) -> float:
        """
        获取骨骼段长度
        
        Args:
            bone_name: 骨骼名称（如 'left_upper_arm'）
        
        Returns:
            长度（米），无效返回0
        """
        bone = self._bones.get(bone_name)
        if bone:
            return bone.length
        return 0.0
    
    def get_distance(self, joint1: str, joint2: str) -> float:
        """
        计算两个关节之间的距离
        
        Args:
            joint1, joint2: 关节名称
        
        Returns:
            距离（米）
        """
        p1 = self.joints.get(joint1)
        p2 = self.joints.get(joint2)
        
        if p1 and p2:
            return p1.distance_to(p2)
        return 0.0
    
    def get_joint_angle(self, angle_name: str) -> float:
        """
        获取关节角度
        
        Args:
            angle_name: 角度名称（如 'left_elbow'）
        
        Returns:
            角度（度）
        """
        if angle_name not in JOINT_ANGLES:
            return 0.0
        
        idx1, idx2, idx3 = JOINT_ANGLES[angle_name]
        name1 = LANDMARK_NAMES.get(idx1)
        name2 = LANDMARK_NAMES.get(idx2)
        name3 = LANDMARK_NAMES.get(idx3)
        
        p1 = self.joints.get(name1)
        p2 = self.joints.get(name2)
        p3 = self.joints.get(name3)
        
        if p1 and p2 and p3:
            return calculate_angle(p1, p2, p3)
        return 0.0
    
    def get_center_of_mass(self) -> Point3D:
        """
        估算质心位置
        
        基于关键关节点的平均位置
        
        Returns:
            质心3D坐标
        """
        key_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        x_sum, y_sum, z_sum = 0.0, 0.0, 0.0
        count = 0
        
        for name in key_joints:
            joint = self.joints.get(name)
            if joint and joint.is_valid():
                x_sum += joint.x
                y_sum += joint.y
                z_sum += joint.z
                count += 1
        
        if count == 0:
            return Point3D(0, 0, 0, confidence=0)
        
        return Point3D(
            x=x_sum / count,
            y=y_sum / count,
            z=z_sum / count,
            confidence=1.0
        )
    
    def get_shoulder_center(self) -> Point3D:
        """获取肩部中心点"""
        left = self.joints.get('left_shoulder')
        right = self.joints.get('right_shoulder')
        
        if left and right:
            return Point3D(
                x=(left.x + right.x) / 2,
                y=(left.y + right.y) / 2,
                z=(left.z + right.z) / 2,
                confidence=min(left.confidence, right.confidence)
            )
        return Point3D(0, 0, 0, confidence=0)
    
    def get_hip_center(self) -> Point3D:
        """获取髋部中心点"""
        left = self.joints.get('left_hip')
        right = self.joints.get('right_hip')
        
        if left and right:
            return Point3D(
                x=(left.x + right.x) / 2,
                y=(left.y + right.y) / 2,
                z=(left.z + right.z) / 2,
                confidence=min(left.confidence, right.confidence)
            )
        return Point3D(0, 0, 0, confidence=0)
    
    def get_overall_confidence(self) -> float:
        """获取整体置信度"""
        if not self.joints:
            return 0.0
        
        confidences = [j.confidence for j in self.joints.values()]
        return sum(confidences) / len(confidences)
    
    def get_valid_joints_count(self, min_confidence: float = 0.5) -> int:
        """获取有效关节数量"""
        return sum(
            1 for j in self.joints.values()
            if j.confidence >= min_confidence and j.is_valid()
        )
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'joints': {
                name: {
                    'x': p.x, 'y': p.y, 'z': p.z,
                    'confidence': p.confidence
                }
                for name, p in self.joints.items()
            },
            'bones': {
                name: {
                    'length': bone.length,
                    'confidence': bone.confidence
                }
                for name, bone in self._bones.items()
            }
        }
    
    def to_array(self) -> np.ndarray:
        """
        转换为numpy数组
        
        Returns:
            shape (33, 4) 的数组，每行为 [x, y, z, confidence]
        """
        arr = np.zeros((33, 4))
        
        for idx, name in LANDMARK_NAMES.items():
            if name in self.joints:
                joint = self.joints[name]
                arr[idx] = [joint.x, joint.y, joint.z, joint.confidence]
        
        return arr
    
    @property
    def bones(self) -> Dict[str, Bone]:
        """所有骨骼段"""
        return self._bones
    
    @property
    def connections(self) -> List[Tuple[str, str]]:
        """骨骼连接列表（用于绘制）"""
        connections = []
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_name = LANDMARK_NAMES.get(start_idx)
            end_name = LANDMARK_NAMES.get(end_idx)
            if start_name and end_name:
                connections.append((start_name, end_name))
        return connections
