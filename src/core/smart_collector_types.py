"""
智能数据采集器类型定义

包含状态枚举、数据类等核心类型

**Feature: smart-data-collector**
**Validates: Requirements 1.1, 2.1, 3.5, 4.1**
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import json


class CollectorState(Enum):
    """
    采集器状态枚举
    
    状态转换:
    IDLE → MONITORING → READY → CAPTURING → PROCESSING → COMPLETED/FAILED
    """
    IDLE = "idle"                   # 空闲，等待开始
    MONITORING = "monitoring"       # 监测稳定性
    READY = "ready"                 # 就绪，等待采集
    CAPTURING = "capturing"         # 采集中
    PROCESSING = "processing"       # 处理中（融合）
    COMPLETED = "completed"         # 完成
    FAILED = "failed"               # 失败


@dataclass
class StabilityResult:
    """
    稳定性检测结果

    Attributes:
        is_stable: 是否稳定
        progress: 稳定进度 0.0-1.0
        body_stable: 身体是否稳定
        left_hand_stable: 左手是否稳定
        right_hand_stable: 右手是否稳定
        body_movement: 身体移动量（米）
        hand_movement: 手部移动量（米）
        stable_frames: 连续稳定帧数
        measurement_stable: 测量数值是否稳定
    """
    is_stable: bool = False
    progress: float = 0.0
    body_stable: bool = False
    left_hand_stable: bool = False
    right_hand_stable: bool = False
    body_movement: float = 0.0
    hand_movement: float = 0.0
    stable_frames: int = 0
    measurement_stable: bool = False

    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StabilityResult':
        """从字典创建"""
        return cls(**data)


@dataclass
class QualityResult:
    """
    质量评分结果
    
    Attributes:
        score: 综合质量分 0.0-1.0
        confidence_score: 置信度分数
        stability_score: 稳定性分数
        completeness_score: 完整性分数
        recommendation: 建议 'auto_save', 'confirm', 'retry'
        issues: 问题列表
        anatomy_valid: 解剖学验证是否通过
        anatomy_issues: 解剖学问题列表
    """
    score: float = 0.0
    confidence_score: float = 0.0
    stability_score: float = 0.0
    completeness_score: float = 0.0
    recommendation: str = 'retry'
    issues: List[str] = field(default_factory=list)
    anatomy_valid: bool = True
    anatomy_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityResult':
        """从字典创建"""
        return cls(**data)


@dataclass
class FusionResult:
    """
    多帧融合结果
    
    Attributes:
        body_measurement: 身体测量结果（字典形式）
        left_hand: 左手测量结果（字典形式）
        right_hand: 右手测量结果（字典形式）
        confidence: 融合后置信度
        frames_used: 使用的帧数
        outliers_removed: 移除的异常值数量
        fusion_timestamp: 融合时间戳
    """
    body_measurement: Optional[Dict] = None
    left_hand: Optional[Dict] = None
    right_hand: Optional[Dict] = None
    confidence: float = 0.0
    frames_used: int = 0
    outliers_removed: int = 0
    fusion_timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FusionResult':
        """从字典创建"""
        return cls(**data)


@dataclass
class CollectorStatus:
    """
    采集器状态信息
    
    Attributes:
        state: 当前状态
        stability_progress: 稳定性进度 0.0-1.0
        quality_score: 质量分数
        message: 状态消息
        saved_path: 保存的文件路径（如果已保存）
        region_status: 各区域稳定状态
    """
    state: CollectorState = CollectorState.IDLE
    stability_progress: float = 0.0
    quality_score: float = 0.0
    message: str = ""
    saved_path: Optional[str] = None
    region_status: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['state'] = self.state.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollectorStatus':
        """从字典创建"""
        data = data.copy()
        if isinstance(data.get('state'), str):
            data['state'] = CollectorState(data['state'])
        return cls(**data)


@dataclass
class SessionData:
    """
    会话数据
    
    Attributes:
        session_id: 会话唯一ID
        start_time: 开始时间
        end_time: 结束时间
        measurements: 测量结果列表
        metadata: 元数据
        summary: 会话摘要
    """
    session_id: str = ""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    measurements: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """从字典创建"""
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SessionData':
        """从JSON字符串创建"""
        return cls.from_dict(json.loads(json_str))


@dataclass 
class CollectorConfig:
    """
    采集器配置
    
    Attributes:
        stability_window: 稳定性检测窗口大小（帧数）
        body_threshold: 身体稳定阈值（米）
        hand_threshold: 手部稳定阈值（米）
        fusion_frames: 融合帧数
        auto_save_threshold: 自动保存质量阈值
        confirm_threshold: 需确认质量阈值
        confidence_weight: 置信度权重
        stability_weight: 稳定性权重
        completeness_weight: 完整性权重
        output_dir: 输出目录
    """
    stability_window: int = 10
    body_threshold: float = 0.02
    hand_threshold: float = 0.01
    fusion_frames: int = 5
    auto_save_threshold: float = 0.8
    confirm_threshold: float = 0.6
    confidence_weight: float = 0.5
    stability_weight: float = 0.4
    completeness_weight: float = 0.1
    output_dir: str = "data/sessions"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollectorConfig':
        """从字典创建"""
        # 只使用已知字段
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
