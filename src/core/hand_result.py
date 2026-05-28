"""
手部检测结果数据结构
定义 HandResult 和 HolisticResult 数据类
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json

from .pose_estimator import Landmark, PoseResult
from .constants import (
    COMBINED_LANDMARK_INDICES,
    HAND_LANDMARK_COUNT,
    BODY_LANDMARK_COUNT,
    TOTAL_LANDMARK_COUNT,
)


@dataclass
class HandResult:
    """
    单手检测结果
    
    Attributes:
        landmarks: 21个2D关键点列表
        detected: 是否检测到手部
        handedness: 左手或右手 ("left" / "right")
        confidence: 整体置信度 (0-1)
    """
    landmarks: List[Landmark] = field(default_factory=list)
    detected: bool = False
    handedness: str = "unknown"
    confidence: float = 0.0
    
    def get_landmark(self, index: int) -> Optional[Landmark]:
        """获取指定索引的关键点"""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None
    
    def get_visible_landmarks(self, threshold: float = 0.5) -> List[tuple]:
        """获取所有可见的关键点"""
        return [
            (i, lm) for i, lm in enumerate(self.landmarks)
            if lm.visibility >= threshold
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'detected': self.detected,
            'handedness': self.handedness,
            'confidence': self.confidence,
            'landmarks': [
                {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                for lm in self.landmarks
            ] if self.landmarks else []
        }

    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HandResult':
        """从字典创建实例"""
        landmarks = []
        for lm_data in data.get('landmarks', []):
            landmarks.append(Landmark(
                x=lm_data['x'],
                y=lm_data['y'],
                z=lm_data['z'],
                visibility=lm_data.get('visibility', 1.0)
            ))
        
        return cls(
            landmarks=landmarks,
            detected=data.get('detected', False),
            handedness=data.get('handedness', 'unknown'),
            confidence=data.get('confidence', 0.0)
        )
    
    @classmethod
    def empty(cls, handedness: str = "unknown") -> 'HandResult':
        """创建空的手部结果"""
        return cls(
            landmarks=[],
            detected=False,
            handedness=handedness,
            confidence=0.0
        )


@dataclass
class HolisticResult:
    """
    Holistic 检测结果
    包含身体姿态和双手检测结果
    
    Attributes:
        pose: 身体检测结果 (33点)
        left_hand: 左手检测结果 (21点)
        right_hand: 右手检测结果 (21点)
        timestamp: 时间戳
    """
    pose: PoseResult = field(default_factory=PoseResult)
    left_hand: HandResult = field(default_factory=lambda: HandResult.empty("left"))
    right_hand: HandResult = field(default_factory=lambda: HandResult.empty("right"))
    timestamp: float = 0.0
    
    @property
    def body_detected(self) -> bool:
        """是否检测到身体"""
        return self.pose.detected
    
    @property
    def hands_detected(self) -> tuple:
        """返回 (左手检测, 右手检测) 状态"""
        return (self.left_hand.detected, self.right_hand.detected)
    
    @property
    def any_hand_detected(self) -> bool:
        """是否检测到任意一只手"""
        return self.left_hand.detected or self.right_hand.detected
    
    @property
    def both_hands_detected(self) -> bool:
        """是否同时检测到双手"""
        return self.left_hand.detected and self.right_hand.detected
    
    def get_combined_landmarks(self) -> List[Optional[Landmark]]:
        """
        获取合并的75点关键点列表
        
        索引映射:
        - 0-32: 身体关键点
        - 33-53: 左手关键点
        - 54-74: 右手关键点
        
        Returns:
            75个关键点列表，未检测到的位置为 None
        """
        combined = [None] * TOTAL_LANDMARK_COUNT
        
        # 身体关键点 (0-32)
        body_start, body_end = COMBINED_LANDMARK_INDICES['body']
        for i, lm in enumerate(self.pose.landmarks):
            if i < body_end - body_start:
                combined[body_start + i] = lm
        
        # 左手关键点 (33-53)
        if self.left_hand.detected:
            left_start, left_end = COMBINED_LANDMARK_INDICES['left_hand']
            for i, lm in enumerate(self.left_hand.landmarks):
                if i < left_end - left_start:
                    combined[left_start + i] = lm
        
        # 右手关键点 (54-74)
        if self.right_hand.detected:
            right_start, right_end = COMBINED_LANDMARK_INDICES['right_hand']
            for i, lm in enumerate(self.right_hand.landmarks):
                if i < right_end - right_start:
                    combined[right_start + i] = lm
        
        return combined
    
    def to_pose_result(self) -> PoseResult:
        """
        转换为兼容的 PoseResult 格式
        仅返回身体33点，用于向后兼容
        
        Returns:
            PoseResult: 仅包含身体关键点的结果
        """
        return self.pose
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于JSON序列化）"""
        return {
            'timestamp': self.timestamp,
            'body_detected': self.body_detected,
            'body_points': [
                {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                for lm in self.pose.landmarks
            ] if self.pose.landmarks else [],
            'hands': {
                'left': self.left_hand.to_dict(),
                'right': self.right_hand.to_dict()
            }
        }
    
    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HolisticResult':
        """从字典创建实例"""
        # 解析身体关键点
        body_landmarks = []
        for lm_data in data.get('body_points', []):
            body_landmarks.append(Landmark(
                x=lm_data['x'],
                y=lm_data['y'],
                z=lm_data['z'],
                visibility=lm_data.get('visibility', 1.0)
            ))
        
        pose = PoseResult(
            landmarks=body_landmarks,
            detected=data.get('body_detected', len(body_landmarks) > 0),
            timestamp=data.get('timestamp', 0.0)
        )
        
        # 解析手部
        hands_data = data.get('hands', {})
        left_hand = HandResult.from_dict(hands_data.get('left', {}))
        right_hand = HandResult.from_dict(hands_data.get('right', {}))
        
        return cls(
            pose=pose,
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp=data.get('timestamp', 0.0)
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'HolisticResult':
        """从JSON字符串创建实例"""
        data = json.loads(json_str)
        return cls.from_dict(data)
