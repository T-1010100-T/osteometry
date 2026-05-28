"""
智能数据采集器

整合稳定性检测、质量评分、多帧融合、会话管理的主控制器

**Feature: smart-data-collector**
**Validates: Requirements 1.2, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4**
"""
from __future__ import annotations
import time
from typing import Dict, Optional, TYPE_CHECKING

from .smart_collector_types import (
    CollectorState, CollectorStatus, CollectorConfig,
    StabilityResult, QualityResult, FusionResult
)
from .stability_detector import StabilityDetector
from .quality_scorer import QualityScorer
from .frame_fusion import FrameFusion
from .session_manager import SessionManager
from .hand_result import HolisticResult
from ..utils.logger import get_logger

# 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from ..measurement.measurement_engine import MeasurementResult
    from ..measurement.hand_measurement import HandMeasurementResult

logger = get_logger(__name__)


class SmartDataCollector:
    """
    智能数据采集器
    
    通过状态机驱动的采集流程，整合稳定性检测、质量评分、多帧融合
    
    状态转换:
    IDLE → MONITORING → READY → CAPTURING → PROCESSING → COMPLETED/FAILED
    
    Example:
        >>> collector = SmartDataCollector()
        >>> status = collector.process_frame(holistic, body_measurement, left_hand, right_hand)
        >>> if status.state == CollectorState.COMPLETED:
        ...     print(f"数据已保存: {status.saved_path}")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化智能数据采集器
        
        Args:
            config: 配置字典
        """
        # 加载配置
        self._config = CollectorConfig.from_dict(config) if config else CollectorConfig()
        
        # 初始化子组件
        self._stability_detector = StabilityDetector(
            window_size=self._config.stability_window,
            body_threshold=self._config.body_threshold,
            hand_threshold=self._config.hand_threshold
        )
        
        self._quality_scorer = QualityScorer(
            confidence_weight=self._config.confidence_weight,
            stability_weight=self._config.stability_weight,
            completeness_weight=self._config.completeness_weight,
            auto_save_threshold=self._config.auto_save_threshold,
            confirm_threshold=self._config.confirm_threshold
        )
        
        self._frame_fusion = FrameFusion(
            fusion_frames=self._config.fusion_frames
        )
        
        self._session_manager = SessionManager(
            output_dir=self._config.output_dir
        )
        
        # 状态
        self._state = CollectorState.IDLE
        self._ready_start_time: Optional[float] = None
        self._ready_countdown = 3.0  # 就绪后倒计时秒数
        
        # 缓存
        self._last_stability: Optional[StabilityResult] = None
        self._last_quality: Optional[QualityResult] = None
        self._last_holistic: Optional[HolisticResult] = None
        
        logger.info("SmartDataCollector 初始化完成")

    
    def process_frame(
        self,
        holistic_result: HolisticResult,
        body_measurement: Optional[MeasurementResult] = None,
        left_hand: Optional[HandMeasurementResult] = None,
        right_hand: Optional[HandMeasurementResult] = None
    ) -> CollectorStatus:
        """
        处理单帧数据
        
        Args:
            holistic_result: Holistic 检测结果
            body_measurement: 身体测量结果
            left_hand: 左手测量结果
            right_hand: 右手测量结果
        
        Returns:
            CollectorStatus: 采集器状态
        """
        try:
            # 缓存输入
            self._last_holistic = holistic_result
            
            # 根据当前状态处理
            if self._state == CollectorState.IDLE:
                return self._handle_idle(holistic_result)
            
            elif self._state == CollectorState.MONITORING:
                return self._handle_monitoring(holistic_result, body_measurement)
            
            elif self._state == CollectorState.READY:
                return self._handle_ready(holistic_result, body_measurement, left_hand, right_hand)
            
            elif self._state == CollectorState.CAPTURING:
                return self._handle_capturing(holistic_result, body_measurement, left_hand, right_hand)
            
            elif self._state == CollectorState.PROCESSING:
                return self._handle_processing()
            
            else:
                return self.get_status()
                
        except Exception as e:
            logger.error(f"处理帧时出错: {e}")
            return self._handle_error(str(e))
    
    def _handle_idle(self, holistic_result: HolisticResult) -> CollectorStatus:
        """处理空闲状态"""
        # 检测到人体后开始监测
        if holistic_result.pose.detected:
            self._state = CollectorState.MONITORING
            self._stability_detector.reset()
            logger.debug("检测到人体，开始监测稳定性")
        
        return self.get_status()
    
    def _handle_monitoring(
        self,
        holistic_result: HolisticResult,
        body_measurement: Optional[MeasurementResult]
    ) -> CollectorStatus:
        """处理监测状态"""
        # 添加帧到稳定性检测器
        self._stability_detector.add_frame(holistic_result)
        
        # 获取稳定性结果
        stability = self._stability_detector.get_stability()
        self._last_stability = stability
        
        # 检查是否稳定
        if stability.is_stable:
            self._state = CollectorState.READY
            self._ready_start_time = time.time()
            logger.info("用户已稳定，准备采集")
        
        return CollectorStatus(
            state=self._state,
            stability_progress=stability.progress,
            message=self._get_stability_message(stability),
            region_status={
                'body': stability.body_stable,
                'left_hand': stability.left_hand_stable,
                'right_hand': stability.right_hand_stable
            }
        )
    
    def _handle_ready(
        self,
        holistic_result: HolisticResult,
        body_measurement: Optional[MeasurementResult],
        left_hand: Optional[HandMeasurementResult],
        right_hand: Optional[HandMeasurementResult]
    ) -> CollectorStatus:
        """处理就绪状态"""
        # 继续检测稳定性
        self._stability_detector.add_frame(holistic_result)
        stability = self._stability_detector.get_stability()
        self._last_stability = stability
        
        # 如果不再稳定，返回监测状态
        if not stability.is_stable:
            self._state = CollectorState.MONITORING
            self._ready_start_time = None
            logger.debug("用户移动，返回监测状态")
            return self.get_status()
        
        # 检查倒计时
        elapsed = time.time() - self._ready_start_time
        remaining = max(0, self._ready_countdown - elapsed)
        
        if remaining <= 0:
            # 开始采集
            self._state = CollectorState.CAPTURING
            self._frame_fusion.reset()
            logger.info("开始采集数据")
        
        return CollectorStatus(
            state=self._state,
            stability_progress=1.0,
            message=f"保持不动... {remaining:.1f}秒",
            region_status={
                'body': stability.body_stable,
                'left_hand': stability.left_hand_stable,
                'right_hand': stability.right_hand_stable
            }
        )
    
    def _handle_capturing(
        self,
        holistic_result: HolisticResult,
        body_measurement: Optional[MeasurementResult],
        left_hand: Optional[HandMeasurementResult],
        right_hand: Optional[HandMeasurementResult]
    ) -> CollectorStatus:
        """处理采集状态"""
        # 计算当前帧质量
        stability = self._stability_detector.get_stability()
        quality = self._quality_scorer.calculate_score(
            holistic_result=holistic_result,
            stability=stability,
            body_measurement=body_measurement,
            hand_measurement=left_hand or right_hand
        )
        self._last_quality = quality
        
        # 添加帧到融合器（即使无效也保存，用于后续分析）
        body_dict = {}
        if body_measurement:
            body_dict = body_measurement.to_dict()
            if body_measurement.is_valid:
                logger.debug(f"采集身体数据: height={body_measurement.height:.3f}m")
            else:
                logger.debug(f"采集身体数据(低置信): height={body_measurement.height:.3f}m")
        
        left_dict = left_hand.to_dict() if left_hand else {}
        right_dict = right_hand.to_dict() if right_hand else {}
        
        is_ready = self._frame_fusion.add_frame(
            body_data=body_dict,
            left_hand_data=left_dict,
            right_hand_data=right_dict,
            quality_score=quality.score
        )
        
        if is_ready:
            self._state = CollectorState.PROCESSING
            logger.debug(f"采集完成，共 {self._frame_fusion.buffer_size} 帧")
        
        return CollectorStatus(
            state=self._state,
            stability_progress=1.0,
            quality_score=quality.score,
            message=f"采集中... {self._frame_fusion.buffer_size}/{self._config.fusion_frames}"
        )
    
    def _handle_processing(self) -> CollectorStatus:
        """处理融合状态"""
        # 执行融合
        fusion_result = self._frame_fusion.fuse()
        
        # 计算最终质量
        quality = self._last_quality or QualityResult()
        
        # 保存数据
        saved_path = None
        if quality.recommendation in ['auto_save', 'confirm']:
            saved_path = self._session_manager.add_measurement(
                fusion_result=fusion_result,
                quality_result=quality
            )
        
        self._state = CollectorState.COMPLETED
        
        return CollectorStatus(
            state=self._state,
            quality_score=quality.score,
            message=f"采集完成！质量: {quality.score:.1%}",
            saved_path=saved_path
        )
    
    def _handle_error(self, error_msg: str) -> CollectorStatus:
        """处理错误"""
        self._state = CollectorState.FAILED
        logger.error(f"采集失败: {error_msg}")
        
        return CollectorStatus(
            state=CollectorState.FAILED,
            message=f"错误: {error_msg}"
        )
    
    def _get_stability_message(self, stability: StabilityResult) -> str:
        """生成稳定性提示消息"""
        if stability.progress < 0.3:
            return "请保持静止..."
        elif stability.progress < 0.7:
            parts = []
            if not stability.body_stable:
                parts.append("身体")
            if not stability.left_hand_stable:
                parts.append("左手")
            if not stability.right_hand_stable:
                parts.append("右手")
            if parts:
                return f"请稳定: {', '.join(parts)}"
            return "正在检测稳定性..."
        else:
            return "即将开始采集..."
    
    def force_capture(self) -> Optional[str]:
        """
        强制立即采集（跳过稳定性检测）
        
        Returns:
            保存的文件路径
        """
        if self._last_holistic is None:
            logger.warning("没有可用的帧数据")
            return None
        
        # 直接进入采集状态
        self._state = CollectorState.CAPTURING
        self._frame_fusion.reset()
        
        # 添加当前帧
        quality = self._quality_scorer.calculate_score(
            holistic_result=self._last_holistic
        )
        
        self._frame_fusion.add_frame(
            body_data={},
            quality_score=quality.score
        )
        
        # 立即融合
        fusion_result = self._frame_fusion.fuse()
        
        # 保存
        saved_path = self._session_manager.add_measurement(
            fusion_result=fusion_result,
            quality_result=quality
        )
        
        self._state = CollectorState.COMPLETED
        return saved_path
    
    def cancel(self) -> None:
        """取消当前采集"""
        self._state = CollectorState.IDLE
        self._stability_detector.reset()
        self._frame_fusion.reset()
        self._ready_start_time = None
        logger.info("采集已取消")
    
    def reset(self) -> None:
        """重置采集器到空闲状态"""
        self.cancel()
    
    def get_status(self) -> CollectorStatus:
        """获取当前状态"""
        stability = self._last_stability or StabilityResult()
        quality = self._last_quality or QualityResult()
        
        return CollectorStatus(
            state=self._state,
            stability_progress=stability.progress,
            quality_score=quality.score,
            region_status={
                'body': stability.body_stable,
                'left_hand': stability.left_hand_stable,
                'right_hand': stability.right_hand_stable
            }
        )
    
    def start_session(self, metadata: Optional[Dict] = None) -> str:
        """开始新会话"""
        return self._session_manager.start_session(metadata)
    
    def end_session(self) -> Optional[Dict]:
        """结束当前会话"""
        return self._session_manager.end_session()
    
    @property
    def state(self) -> CollectorState:
        """当前状态"""
        return self._state
    
    @property
    def is_collecting(self) -> bool:
        """是否正在采集"""
        return self._state in [
            CollectorState.MONITORING,
            CollectorState.READY,
            CollectorState.CAPTURING,
            CollectorState.PROCESSING
        ]
