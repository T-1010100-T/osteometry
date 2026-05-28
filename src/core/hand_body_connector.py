"""
手部-身体连接器

解决手部精度模型和身体模型的连接问题：
1. 手腕位置对齐 - 将手部手腕对齐到身体手腕
2. 深度一致性 - 确保手部所有点使用一致的深度参考
3. 坐标系融合 - 平滑过渡手部和身体坐标
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .coordinate_transformer import Point3D
from .hand_coordinate_transformer import HandCoordinateTransformer
from .hand_result import HandResult
from .pose_estimator import PoseResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HandBodyConnectionConfig:
    """手部-身体连接配置"""
    # 手腕对齐
    enable_wrist_alignment: bool = True
    wrist_alignment_weight: float = 0.7  # 身体手腕权重 (0-1)
    
    # 深度一致性
    enable_depth_consistency: bool = True
    max_depth_deviation: float = 0.08  # 最大深度偏差 (米)
    
    # 平滑过渡
    enable_smooth_transition: bool = True
    transition_blend_factor: float = 0.3  # 过渡混合因子
    
    # 置信度阈值
    min_body_wrist_visibility: float = 0.5
    min_hand_wrist_confidence: float = 0.3


class HandBodyConnector:
    """
    手部-身体连接器
    
    功能：
    1. 将手部手腕位置对齐到身体手腕
    2. 确保手部深度与身体一致
    3. 平滑过渡手部和身体坐标
    
    Example:
        >>> connector = HandBodyConnector(config)
        >>> aligned_hand_3d = connector.align_hand_to_body(
        ...     hand_3d, body_wrist_3d, hand_wrist_3d
        ... )
    """
    
    def __init__(self, config: Optional[HandBodyConnectionConfig] = None):
        """
        初始化连接器
        
        Args:
            config: 连接配置，None 使用默认配置
        """
        self.config = config or HandBodyConnectionConfig()
    
    def get_body_wrist_3d(
        self,
        body_points_3d: List[Point3D],
        hand_side: str
    ) -> Optional[Point3D]:
        """
        获取身体手腕的3D坐标
        
        Args:
            body_points_3d: 身体3D点列表
            hand_side: "left" 或 "right"
        
        Returns:
            身体手腕3D坐标，或 None
        """
        # 身体手腕索引: 左手=15, 右手=16
        wrist_idx = 15 if hand_side == "left" else 16
        
        if wrist_idx >= len(body_points_3d):
            return None
        
        wrist = body_points_3d[wrist_idx]
        
        # 检查有效性
        if not wrist.is_valid() or wrist.confidence < self.config.min_body_wrist_visibility:
            return None
        
        return wrist
    
    def align_hand_to_body(
        self,
        hand_3d: List[Point3D],
        body_wrist_3d: Point3D,
        hand_wrist_3d: Optional[Point3D] = None
    ) -> List[Point3D]:
        """
        将手部3D点对齐到身体手腕
        
        通过计算手部手腕和身体手腕的偏移，
        将整个手部平移到正确位置。
        
        Args:
            hand_3d: 手部21个3D点
            body_wrist_3d: 身体手腕3D坐标
            hand_wrist_3d: 手部手腕3D坐标（可选，默认使用 hand_3d[0]）
        
        Returns:
            对齐后的手部3D点列表
        """
        if len(hand_3d) != 21:
            return hand_3d
        
        if not body_wrist_3d.is_valid():
            return hand_3d
        
        # 获取手部手腕
        if hand_wrist_3d is None:
            hand_wrist_3d = hand_3d[0]
        
        if not hand_wrist_3d.is_valid():
            return hand_3d
        
        # 计算偏移量
        offset_x = body_wrist_3d.x - hand_wrist_3d.x
        offset_y = body_wrist_3d.y - hand_wrist_3d.y
        offset_z = body_wrist_3d.z - hand_wrist_3d.z
        
        # 应用权重（部分对齐）
        weight = self.config.wrist_alignment_weight
        offset_x *= weight
        offset_y *= weight
        offset_z *= weight
        
        # 平移所有手部点
        aligned_hand = []
        for i, point in enumerate(hand_3d):
            if not point.is_valid():
                aligned_hand.append(point)
                continue
            
            # 手腕点完全对齐
            if i == 0:
                new_point = Point3D(
                    x=body_wrist_3d.x,
                    y=body_wrist_3d.y,
                    z=body_wrist_3d.z,
                    confidence=max(point.confidence, body_wrist_3d.confidence)
                )
            else:
                # 其他点应用偏移
                new_point = Point3D(
                    x=point.x + offset_x,
                    y=point.y + offset_y,
                    z=point.z + offset_z,
                    confidence=point.confidence
                )
            
            aligned_hand.append(new_point)
        
        return aligned_hand
    
    def enforce_depth_consistency(
        self,
        hand_3d: List[Point3D],
        reference_depth: float
    ) -> List[Point3D]:
        """
        强制手部深度一致性
        
        将深度偏差过大的点校正到合理范围内。
        
        Args:
            hand_3d: 手部21个3D点
            reference_depth: 参考深度（通常是手腕深度）
        
        Returns:
            深度校正后的手部3D点列表
        """
        if reference_depth <= 0:
            return hand_3d
        
        max_dev = self.config.max_depth_deviation
        corrected_hand = []
        
        for point in hand_3d:
            if not point.is_valid():
                corrected_hand.append(point)
                continue
            
            depth_diff = point.z - reference_depth
            
            # 检查深度偏差
            if abs(depth_diff) > max_dev:
                # 限制深度偏差
                if depth_diff > 0:
                    new_z = reference_depth + max_dev
                else:
                    new_z = reference_depth - max_dev
                
                # 重新计算 x, y（保持投影位置不变）
                # 由于深度变化，需要按比例调整 x, y
                scale = new_z / point.z if point.z > 0 else 1.0
                
                new_point = Point3D(
                    x=point.x * scale,
                    y=point.y * scale,
                    z=new_z,
                    confidence=point.confidence * 0.9  # 略微降低置信度
                )
                corrected_hand.append(new_point)
            else:
                corrected_hand.append(point)
        
        return corrected_hand
    
    def connect_hand_to_body(
        self,
        hand_3d: List[Point3D],
        body_points_3d: List[Point3D],
        hand_side: str
    ) -> List[Point3D]:
        """
        完整的手部-身体连接流程
        
        Args:
            hand_3d: 手部21个3D点
            body_points_3d: 身体33个3D点
            hand_side: "left" 或 "right"
        
        Returns:
            连接后的手部3D点列表
        """
        if len(hand_3d) != 21:
            logger.debug(f"手部点数量不正确: {len(hand_3d)}")
            return hand_3d
        
        # 获取身体手腕
        body_wrist = self.get_body_wrist_3d(body_points_3d, hand_side)
        if body_wrist is None:
            logger.debug(f"{hand_side} 身体手腕无效，跳过连接")
            return hand_3d
        
        result = hand_3d
        
        # 1. 手腕对齐
        if self.config.enable_wrist_alignment:
            result = self.align_hand_to_body(result, body_wrist)
            logger.debug(f"{hand_side} 手部已对齐到身体手腕")
        
        # 2. 深度一致性
        if self.config.enable_depth_consistency:
            result = self.enforce_depth_consistency(result, body_wrist.z)
        
        return result
    
    def update_body_with_hand(
        self,
        body_points_3d: List[Point3D],
        hand_3d: List[Point3D],
        hand_side: str
    ) -> List[Point3D]:
        """
        用手部精细点更新身体关键点
        
        将手部模型的精细关键点映射到身体模型的手部位置。
        
        MediaPipe 身体手部关键点:
        - 17/18: 左/右小指
        - 19/20: 左/右食指
        - 21/22: 左/右拇指
        
        MediaPipe 手部关键点:
        - 0: 手腕
        - 1-4: 拇指 (1=CMC, 2=MCP, 3=IP, 4=TIP)
        - 5-8: 食指 (5=MCP, 6=PIP, 7=DIP, 8=TIP)
        - 9-12: 中指
        - 13-16: 无名指
        - 17-20: 小指 (17=MCP, 18=PIP, 19=DIP, 20=TIP)
        
        Args:
            body_points_3d: 身体33个3D点
            hand_3d: 手部21个3D点（已对齐）
            hand_side: "left" 或 "right"
        
        Returns:
            更新后的身体3D点列表
        """
        if len(hand_3d) != 21 or len(body_points_3d) < 33:
            return body_points_3d
        
        # 复制身体点
        updated_body = list(body_points_3d)
        
        # 映射关系
        if hand_side == "left":
            # 左手
            mappings = [
                (17, 17),  # 身体左小指 <- 手部小指MCP
                (19, 5),   # 身体左食指 <- 手部食指MCP
                (21, 1),   # 身体左拇指 <- 手部拇指CMC
            ]
        else:
            # 右手
            mappings = [
                (18, 17),  # 身体右小指 <- 手部小指MCP
                (20, 5),   # 身体右食指 <- 手部食指MCP
                (22, 1),   # 身体右拇指 <- 手部拇指CMC
            ]
        
        for body_idx, hand_idx in mappings:
            if hand_3d[hand_idx].is_valid():
                updated_body[body_idx] = hand_3d[hand_idx]
        
        return updated_body


def create_hand_body_connector(
    config_dict: Optional[dict] = None
) -> HandBodyConnector:
    """
    创建手部-身体连接器
    
    Args:
        config_dict: 配置字典，可选
    
    Returns:
        HandBodyConnector 实例
    """
    if config_dict is None:
        return HandBodyConnector()
    
    config = HandBodyConnectionConfig(
        enable_wrist_alignment=config_dict.get('enable_wrist_alignment', True),
        wrist_alignment_weight=config_dict.get('wrist_alignment_weight', 0.7),
        enable_depth_consistency=config_dict.get('enable_depth_consistency', True),
        max_depth_deviation=config_dict.get('max_depth_deviation', 0.08),
        enable_smooth_transition=config_dict.get('enable_smooth_transition', True),
        transition_blend_factor=config_dict.get('transition_blend_factor', 0.3),
        min_body_wrist_visibility=config_dict.get('min_body_wrist_visibility', 0.5),
        min_hand_wrist_confidence=config_dict.get('min_hand_wrist_confidence', 0.3),
    )
    
    return HandBodyConnector(config)
