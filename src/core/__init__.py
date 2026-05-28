"""核心算法层"""
from .constants import (
    PoseLandmark, LANDMARK_NAMES, POSE_CONNECTIONS,
    HandLandmark, HAND_LANDMARK_NAMES, HAND_CONNECTIONS,
    COMBINED_LANDMARK_INDICES, BODY_LANDMARK_COUNT, HAND_LANDMARK_COUNT, TOTAL_LANDMARK_COUNT
)
from .pose_estimator import PoseEstimator, PoseEstimatorCompat, PoseResult, Landmark
from .coordinate_transformer import CoordinateTransformer, Point3D, calculate_angle, calculate_distance
from .keypoint_filter import KeypointFilter, KeypointFilterAdvanced
from .skeleton import Skeleton3D, Bone
from .hand_result import HandResult, HolisticResult
from .holistic_estimator import HolisticEstimator
from .hand_skeleton import HandSkeleton3D
from .hand_coordinate_transformer import HandCoordinateTransformer
from .hand_filter import HandKeypointFilter, HandTracker
from .gesture_recognizer import GestureRecognizer, GestureType, GestureResult
from .roi_optimizer import ROIOptimizer, ROI
from .hand_exceptions import (
    HandDetectionError, HandModelLoadError, HandDepthError,
    HandTrackingLostError, GestureRecognitionError, HandCoordinateError, GPUMemoryError
)
# 智能数据采集器
from .smart_collector_types import (
    CollectorState, StabilityResult, QualityResult, FusionResult,
    CollectorStatus, SessionData, CollectorConfig
)
from .stability_detector import StabilityDetector
from .quality_scorer import QualityScorer
from .frame_fusion import FrameFusion
from .session_manager import SessionManager
from .smart_collector import SmartDataCollector

# 相机和测量模块
from .camera_controller import CameraController, CameraConfig, CameraState
from .measurement_collector import MeasurementCollector, CollectorConfig as MeasCollectorConfig, CaptureState
from .data_aggregator import (
    REQUIRED_MEASUREMENT_FIELDS_CN, FIELD_MAX_CM,
    to_cm_or_none, validate_measurement_value_cm, mean_drop_min_max,
    extract_required_fields, aggregate_samples, format_measurement_txt
)
from .measurement_engine import MeasurementEngine

# 深度优化模块
from .depth_config import (
    FilterChainConfig, SamplerConfig, DepthProcessorConfig,
    BODY_PART_SAMPLING_STRATEGIES, KEYPOINT_ID_TO_BODY_PART,
    get_sampling_strategy, load_depth_config
)
from .adaptive_depth_sampler import AdaptiveDepthSampler
from .depth_processor import DepthProcessor

# 关键点稳定模块
from .stabilizer_config import (
    OneEuroConfig, StabilizerConfig,
    STABILIZER_PRESETS, get_stabilizer_preset
)
from .keypoint_stabilizer import (
    LowPassFilter, OneEuroFilter, KeypointStabilizer
)

# 生物力学约束模块
from .biomechanical_constraints import (
    ConstraintsConfig, BiomechanicalConstraints,
    PROPORTION_CONSTRAINTS, BONE_NAME_MAPPING, SYMMETRIC_BONE_PAIRS,
    ProportionResult, SymmetryResult, ConstraintsResult
)

# 手部-身体连接模块
from .hand_body_connector import (
    HandBodyConnector, HandBodyConnectionConfig, create_hand_body_connector
)

# 人体分割模块
from .body_segmentation import BodySegmentation, SegmentationConfig

__all__ = [
    # 身体常量
    "PoseLandmark", "LANDMARK_NAMES", "POSE_CONNECTIONS",
    # 手部常量
    "HandLandmark", "HAND_LANDMARK_NAMES", "HAND_CONNECTIONS",
    "COMBINED_LANDMARK_INDICES", "BODY_LANDMARK_COUNT", "HAND_LANDMARK_COUNT", "TOTAL_LANDMARK_COUNT",
    # 估计器
    "PoseEstimator", "PoseEstimatorCompat", "PoseResult", "Landmark",
    "HolisticEstimator", "HolisticResult", "HandResult",
    # 坐标变换
    "CoordinateTransformer", "Point3D", "calculate_angle", "calculate_distance",
    # 滤波
    "KeypointFilter", "KeypointFilterAdvanced",
    "HandKeypointFilter", "HandTracker",
    # 骨骼模型
    "Skeleton3D", "Bone",
    "HandSkeleton3D",
    # 手部坐标变换
    "HandCoordinateTransformer",
    # 手势识别
    "GestureRecognizer", "GestureType", "GestureResult",
    # ROI优化
    "ROIOptimizer", "ROI",
    # 异常类
    "HandDetectionError", "HandModelLoadError", "HandDepthError",
    "HandTrackingLostError", "GestureRecognitionError", "HandCoordinateError", "GPUMemoryError",
    # 智能数据采集器
    "SmartDataCollector", "CollectorState", "CollectorStatus", "CollectorConfig",
    "StabilityDetector", "StabilityResult",
    "QualityScorer", "QualityResult",
    "FrameFusion", "FusionResult",
    "SessionManager", "SessionData",
    # 相机和测量模块
    "CameraController", "CameraConfig", "CameraState",
    "MeasurementCollector", "MeasCollectorConfig", "CaptureState",
    "MeasurementEngine",
    "REQUIRED_MEASUREMENT_FIELDS_CN", "FIELD_MAX_CM",
    "to_cm_or_none", "validate_measurement_value_cm", "mean_drop_min_max",
    "extract_required_fields", "aggregate_samples", "format_measurement_txt",
    # 深度优化模块
    "FilterChainConfig", "SamplerConfig", "DepthProcessorConfig",
    "BODY_PART_SAMPLING_STRATEGIES", "KEYPOINT_ID_TO_BODY_PART",
    "get_sampling_strategy", "load_depth_config",
    "AdaptiveDepthSampler",
    "DepthProcessor",
    # 关键点稳定模块
    "OneEuroConfig", "StabilizerConfig",
    "STABILIZER_PRESETS", "get_stabilizer_preset",
    "LowPassFilter", "OneEuroFilter", "KeypointStabilizer",
    # 生物力学约束模块
    "ConstraintsConfig", "BiomechanicalConstraints",
    "PROPORTION_CONSTRAINTS", "BONE_NAME_MAPPING", "SYMMETRIC_BONE_PAIRS",
    "ProportionResult", "SymmetryResult", "ConstraintsResult",
    # 手部-身体连接模块
    "HandBodyConnector", "HandBodyConnectionConfig", "create_hand_body_connector",
    # 人体分割模块
    "BodySegmentation", "SegmentationConfig",
]
