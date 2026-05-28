"""
数据聚合模块

测量数据的验证、过滤和融合
"""
import math
from typing import Dict, List, Optional, Tuple


# 必需的测量字段（中文）
REQUIRED_MEASUREMENT_FIELDS_CN = [
    '身高',
    '坐高',
    '肩宽',
    '骨盆宽',
    '臂展',
    '上肢长',
    '下肢长',
    '腿长',
    '颈臀长',
    '手长',
    '足长',
    '脊柱底部到中部',
    '脊柱中部到肩部',
    '脊柱肩部到头部',
    '脊柱肩部到左肩',
    '左肩到左肘',
    '左肘到左腕',
    '左腕到左手',
    '脊柱肩部到右肩',
    '右肩到右肘',
    '右肘到右腕',
    '右腕到右手',
    '脊柱底部到左髋',
    '左髋到左膝',
    '左膝到左踝',
    '左踝到左脚',
    '脊柱底部到右髋',
    '右髋到右膝',
    '右膝到右踝',
    '右踝到右脚',
]

# 各字段的最大合理值（cm）
FIELD_MAX_CM: Dict[str, float] = {
    '身高': 250.0,
    '坐高': 150.0,
    '肩宽': 80.0,
    '骨盆宽': 60.0,
    '臂展': 280.0,
    '上肢长': 100.0,
    '下肢长': 120.0,
    '腿长': 120.0,
    '颈臀长': 100.0,
    '手长': 30.0,
    '足长': 40.0,
    '脊柱底部到中部': 80.0,
    '脊柱中部到肩部': 80.0,
    '脊柱肩部到头部': 60.0,
    '脊柱肩部到左肩': 40.0,
    '脊柱肩部到右肩': 40.0,
    '左肩到左肘': 55.0,
    '右肩到右肘': 55.0,
    '左肘到左腕': 55.0,
    '右肘到右腕': 55.0,
    '左腕到左手': 30.0,
    '右腕到右手': 30.0,
    '脊柱底部到左髋': 35.0,
    '脊柱底部到右髋': 35.0,
    '左髋到左膝': 70.0,
    '右髋到右膝': 70.0,
    '左膝到左踝': 70.0,
    '右膝到右踝': 70.0,
    '左踝到左脚': 30.0,
    '右踝到右脚': 30.0,
}


def to_cm_or_none(value_m: Optional[float], scale: float = 1.0) -> Optional[float]:
    """将米转换为厘米，无效值返回 None"""
    if value_m is None:
        return None
    try:
        v = float(value_m)
    except Exception:
        return None
    if not math.isfinite(v) or v <= 0.0:
        return None
    return v * 100.0 * float(scale)


def validate_measurement_value_cm(key: str, value_cm) -> Optional[float]:
    """验证测量值是否在合理范围内"""
    if value_cm is None:
        return None
    try:
        v = float(value_cm)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    if v <= 0.0:
        return None
    max_v = FIELD_MAX_CM.get(key)
    if max_v is not None and v > float(max_v):
        return None
    return v


def mean_drop_min_max(values: List[float]) -> float:
    """去掉最大最小值后取平均"""
    if not values:
        return 0.0
    vals = [float(v) for v in values]
    if len(vals) <= 2:
        return float(sum(vals) / len(vals))
    vals.sort()
    trimmed = vals[1:-1]
    if not trimmed:
        return float(sum(vals) / len(vals))
    return float(sum(trimmed) / len(trimmed))


def extract_required_fields(measurement_values_cm: Optional[Dict]) -> Optional[Dict[str, float]]:
    """从测量值中提取必需字段"""
    if not measurement_values_cm:
        return None
    result: Dict[str, float] = {}
    for k in REQUIRED_MEASUREMENT_FIELDS_CN:
        v = measurement_values_cm.get(k, None)
        vv = validate_measurement_value_cm(k, v)
        if vv is None:
            continue
        result[k] = vv
    return result


def aggregate_samples(samples: List[Dict[str, float]]) -> Tuple[Dict[str, Optional[float]], Dict[str, int]]:
    """
    聚合多帧采样数据
    
    Returns:
        (aggregated_values, sample_counts)
    """
    aggregated: Dict[str, Optional[float]] = {}
    counts: Dict[str, int] = {}
    
    for k in REQUIRED_MEASUREMENT_FIELDS_CN:
        vals = [s.get(k) for s in samples if s is not None and k in s and s.get(k) is not None]
        counts[k] = len(vals)
        if not vals:
            aggregated[k] = None
        else:
            aggregated[k] = mean_drop_min_max(vals)
    
    return aggregated, counts


def format_measurement_txt(meta: Dict, aggregated_cm: Dict[str, float]) -> str:
    """格式化测量结果为文本"""
    lines: List[str] = []
    lines.append("=" * 50)
    lines.append("        身体测量数据（单位：cm）")
    lines.append("=" * 50)
    lines.append("")
    
    ts = meta.get("timestamp", "")
    camera_mode = meta.get("camera_mode", "")
    lines.append(f"测量时间\t{ts}")
    lines.append(f"相机模式\t{camera_mode}")
    lines.append(f"站稳帧阈值\t{meta.get('stable_frames_required', '')}")
    lines.append(f"倒计时\t{meta.get('countdown_seconds', '')}s")
    lines.append(f"采样窗口\t{meta.get('sampling_window_seconds', '')}s")
    lines.append(f"目标采样帧数\t{meta.get('sampling_target_frames', '')}")
    lines.append(f"实际采样帧数\t{meta.get('sampling_frames_collected', '')}")
    lines.append(f"融合规则\t去最大/最小后取平均")
    lines.append("")
    lines.append("-" * 50)
    
    for k in REQUIRED_MEASUREMENT_FIELDS_CN:
        v = aggregated_cm.get(k, None)
        if v is None:
            lines.append(f"{k}\tN/A")
            continue
        try:
            v = float(v)
        except Exception:
            v = 0.0
        lines.append(f"{k}\t{v:.1f}cm")
    
    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)
