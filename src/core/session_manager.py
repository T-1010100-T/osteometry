"""
会话管理器

管理采集会话的生命周期和数据组织

**Feature: smart-data-collector**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .smart_collector_types import SessionData, FusionResult, QualityResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


# 字段名称中英文映射（用于生成易读表格）
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

# 需要输出的字段顺序（按类别分组）
EXPORT_FIELD_ORDER = [
    # 基本尺寸
    'height',
    'shoulder_width',
    'hip_width',
    'arm_span',
    'torso_length',
    'neck_length',
    
    # Bike Fitting 躯干
    'spine_base_to_spine_mid',
    'spine_mid_to_spine_shoulder',
    'spine_shoulder_to_head',
    
    # Bike Fitting 左上肢
    'spine_shoulder_to_shoulder_left',
    'shoulder_left_to_elbow_left',
    'elbow_left_to_wrist_left',
    'wrist_left_to_hand_left',
    
    # Bike Fitting 右上肢
    'spine_shoulder_to_shoulder_right',
    'shoulder_right_to_elbow_right',
    'elbow_right_to_wrist_right',
    'wrist_right_to_hand_right',
    
    # Bike Fitting 左下肢
    'spine_base_to_hip_left',
    'hip_left_to_knee_left',
    'knee_left_to_ankle_left',
    'ankle_left_to_foot_left',
    
    # Bike Fitting 右下肢
    'spine_base_to_hip_right',
    'hip_right_to_knee_right',
    'knee_right_to_ankle_right',
    'ankle_right_to_foot_right',
]


class SessionManager:
    """
    会话管理器
    
    管理测量会话的创建、数据存储和历史查询
    
    Example:
        >>> manager = SessionManager(output_dir='data/sessions')
        >>> session_id = manager.start_session()
        >>> manager.add_measurement(fusion_result, quality_result)
        >>> summary = manager.end_session()
    """
    
    def __init__(
        self,
        output_dir: str = 'data/sessions',
        auto_save: bool = True
    ):
        """
        初始化会话管理器
        
        Args:
            output_dir: 输出目录
            auto_save: 是否自动保存
        """
        self.output_dir = Path(output_dir)
        self.auto_save = auto_save
        
        self._current_session: Optional[SessionData] = None
        self._session_file: Optional[Path] = None
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"SessionManager 初始化: output_dir={output_dir}")
    
    def start_session(self, metadata: Optional[Dict] = None) -> str:
        """
        开始新会话
        
        Args:
            metadata: 会话元数据
        
        Returns:
            会话ID
        """
        # 生成唯一会话ID
        timestamp = datetime.now()
        session_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self._current_session = SessionData(
            session_id=session_id,
            start_time=timestamp.isoformat(),
            metadata=metadata or {},
            measurements=[]
        )
        
        # 创建会话文件
        self._session_file = self.output_dir / f"{session_id}.json"
        
        if self.auto_save:
            self._save_session()
        
        logger.info(f"会话已开始: {session_id}")
        return session_id

    
    def add_measurement(
        self,
        fusion_result: FusionResult,
        quality_result: Optional[QualityResult] = None,
        measurement_type: str = 'full_body'
    ) -> str:
        """
        添加测量结果到当前会话
        
        Args:
            fusion_result: 融合结果
            quality_result: 质量评分结果
            measurement_type: 测量类型
        
        Returns:
            保存的文件路径
        """
        if self._current_session is None:
            # 自动开始新会话
            self.start_session()
        
        # 构建测量记录
        measurement = {
            'timestamp': datetime.now().isoformat(),
            'type': measurement_type,
            'quality_score': quality_result.score if quality_result else 0.0,
            'recommendation': quality_result.recommendation if quality_result else 'unknown',
            'body': fusion_result.body_measurement,
            'left_hand': fusion_result.left_hand,
            'right_hand': fusion_result.right_hand,
            'frames_used': fusion_result.frames_used,
            'outliers_removed': fusion_result.outliers_removed,
            'confidence': fusion_result.confidence
        }
        
        # 添加解剖学验证结果
        if quality_result:
            measurement['anatomy_valid'] = quality_result.anatomy_valid
            measurement['anatomy_issues'] = quality_result.anatomy_issues
        
        self._current_session.measurements.append(measurement)
        
        if self.auto_save:
            self._save_session()
        
        # 生成文件名：YYYY-MM-DD HH-MM-SS 格式
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        
        # JSON 文件（保留详细信息用于程序读取）
        json_file = self.output_dir / f"{timestamp_str}.json"
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(measurement, f, indent=2, ensure_ascii=False)
            logger.info(f"测量数据已保存: {json_file}")
            
            # 同时生成易读的 TXT 报告
            txt_file = self.output_dir / f"{timestamp_str}.txt"
            self._export_to_txt(measurement, txt_file)
            
        except Exception as e:
            logger.error(f"保存测量数据失败: {e}")
        
        return str(json_file)
    
    def end_session(self) -> Optional[Dict]:
        """
        结束当前会话
        
        Returns:
            会话摘要
        """
        if self._current_session is None:
            logger.warning("没有活动会话")
            return None
        
        # 设置结束时间
        self._current_session.end_time = datetime.now().isoformat()
        
        # 计算会话摘要
        summary = self._calculate_summary()
        self._current_session.summary = summary
        
        if self.auto_save:
            self._save_session()
        
        logger.info(f"会话已结束: {self._current_session.session_id}")
        
        result = summary
        self._current_session = None
        self._session_file = None
        
        return result
    
    def _calculate_summary(self) -> Dict:
        """计算会话摘要统计"""
        if not self._current_session or not self._current_session.measurements:
            return {}
        
        measurements = self._current_session.measurements
        
        # 收集质量分数
        quality_scores = [m.get('quality_score', 0) for m in measurements]
        
        # 收集身体测量数据
        heights = []
        for m in measurements:
            if m.get('body') and 'height' in m['body']:
                heights.append(m['body']['height'])
        
        summary = {
            'total_measurements': len(measurements),
            'quality_stats': {
                'mean': float(np.mean(quality_scores)) if quality_scores else 0,
                'std': float(np.std(quality_scores)) if quality_scores else 0,
                'min': float(np.min(quality_scores)) if quality_scores else 0,
                'max': float(np.max(quality_scores)) if quality_scores else 0
            },
            'anatomy_valid_count': sum(1 for m in measurements if m.get('anatomy_valid', True)),
            'duration_seconds': self._calculate_duration()
        }
        
        if heights:
            summary['height_stats'] = {
                'mean': float(np.mean(heights)),
                'std': float(np.std(heights)),
                'min': float(np.min(heights)),
                'max': float(np.max(heights))
            }
        
        return summary
    
    def _calculate_duration(self) -> float:
        """计算会话持续时间（秒）"""
        if not self._current_session:
            return 0.0
        
        try:
            start = datetime.fromisoformat(self._current_session.start_time)
            end = datetime.fromisoformat(self._current_session.end_time) if self._current_session.end_time else datetime.now()
            return (end - start).total_seconds()
        except:
            return 0.0
    
    def _save_session(self) -> None:
        """保存会话到文件"""
        if not self._current_session or not self._session_file:
            return
        
        try:
            with open(self._session_file, 'w', encoding='utf-8') as f:
                json.dump(self._current_session.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存会话失败: {e}")
    
    def _export_to_txt(self, measurement: Dict, txt_file: Path) -> None:
        """
        将测量数据导出为易读的 TXT 表格
        
        Args:
            measurement: 测量数据字典
            txt_file: TXT 文件路径
        """
        body = measurement.get('body')
        if not body:
            logger.warning("无身体测量数据，跳过导出")
            return
        
        try:
            lines = []
            
            # 标题
            lines.append("=" * 40)
            lines.append("        身体测量数据报告")
            lines.append("=" * 40)
            lines.append("")
            
            # 数据行：使用制表符对齐
            for field in EXPORT_FIELD_ORDER:
                if field in body:
                    value = body[field]
                    if isinstance(value, (int, float)) and value > 0:
                        mm_value = f"{value * 1000:.1f}mm"
                    else:
                        mm_value = "0.0mm"
                    
                    cn_name = FIELD_NAMES_CN.get(field, field)
                    lines.append(f"{cn_name}\t{mm_value}")
            
            # 元数据
            lines.append("")
            lines.append("-" * 40)
            lines.append("测量信息")
            lines.append("-" * 40)
            
            timestamp = measurement.get('timestamp', '')[:19]
            quality = f"{measurement.get('quality_score', 0):.1%}"
            frames = str(measurement.get('frames_used', 0))
            confidence = f"{measurement.get('confidence', 0):.1%}"
            
            lines.append(f"测量时间\t{timestamp}")
            lines.append(f"质量分数\t{quality}")
            lines.append(f"融合帧数\t{frames}")
            lines.append(f"置信度\t{confidence}")
            
            # 数据质量问题
            if measurement.get('anatomy_issues'):
                lines.append("")
                lines.append("-" * 40)
                lines.append("数据质量问题")
                lines.append("-" * 40)
                for issue in measurement['anatomy_issues']:
                    lines.append(f"• {issue}")
            
            lines.append("")
            lines.append("=" * 40)
            
            # 写入文件
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.info(f"测量报告已保存: {txt_file}")
            
        except Exception as e:
            logger.error(f"导出报告失败: {e}")
    
    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """
        获取历史会话列表
        
        Args:
            limit: 返回数量限制
        
        Returns:
            会话摘要列表
        """
        sessions = []
        
        # 查找所有会话文件
        session_files = sorted(
            self.output_dir.glob('*.json'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for file_path in session_files[:limit]:
            try:
                # 只读取包含 session_id 的文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'session_id' in data:
                        sessions.append({
                            'session_id': data.get('session_id'),
                            'start_time': data.get('start_time'),
                            'end_time': data.get('end_time'),
                            'measurement_count': len(data.get('measurements', [])),
                            'summary': data.get('summary')
                        })
            except Exception as e:
                logger.debug(f"读取会话文件失败: {file_path}, {e}")
        
        return sessions
    
    def load_session(self, session_id: str) -> Optional[SessionData]:
        """
        加载指定会话
        
        Args:
            session_id: 会话ID
        
        Returns:
            会话数据
        """
        session_file = self.output_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.warning(f"会话文件不存在: {session_file}")
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return SessionData.from_dict(data)
        except Exception as e:
            logger.error(f"加载会话失败: {e}")
            return None
    
    @property
    def current_session_id(self) -> Optional[str]:
        """当前会话ID"""
        return self._current_session.session_id if self._current_session else None
    
    @property
    def has_active_session(self) -> bool:
        """是否有活动会话"""
        return self._current_session is not None
