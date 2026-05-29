"""
测量引擎模块

整合骨骼检测、坐标变换和测量计算

优化功能：
- 可选的深度处理器（DepthProcessor）进行深度图增强
- 可选的关键点稳定器（KeypointStabilizer）减少抖动
- 人体深度分割，排除背景干扰
- 自适应深度采样提高测量精度
"""
import os
import time
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING

import numpy as np

from src.core.skeleton import Skeleton3D
from src.core.coordinate_transformer import CoordinateTransformer, Point3D
from src.core.hand_coordinate_transformer import HandCoordinateTransformer
from src.core.body_segmentation import BodySegmentation, SegmentationConfig
from src.core.data_aggregator import to_cm_or_none

if TYPE_CHECKING:
    from src.core.depth_processor import DepthProcessor
    from src.core.keypoint_stabilizer import KeypointStabilizer
    from src.core.biomechanical_constraints import BiomechanicalConstraints, ConstraintsResult


def _env_float(name: str, default: float) -> float:
    """从环境变量获取浮点数"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """从环境变量获取布尔值"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ('1', 'true', 'yes', 'on')


class MeasurementEngine:
    """
    测量引擎
    
    整合骨骼检测、坐标变换和测量计算
    
    优化功能：
    - 可选的深度处理器进行深度图增强
    - 可选的关键点稳定器减少抖动
    - 可选的生物力学约束验证
    - 人体深度分割，排除背景干扰
    - 自适应深度采样提高测量精度
    """
    
    def __init__(
        self,
        transformer: CoordinateTransformer, 
        hand_transformer: Optional[HandCoordinateTransformer] = None,
        depth_processor: Optional['DepthProcessor'] = None,
        keypoint_stabilizer: Optional['KeypointStabilizer'] = None,
        biomechanical_constraints: Optional['BiomechanicalConstraints'] = None
    ):
        """
        初始化测量引擎
        
        Args:
            transformer: 坐标变换器
            hand_transformer: 手部坐标变换器（可选）
            depth_processor: 深度处理器（可选，用于深度图增强）
            keypoint_stabilizer: 关键点稳定器（可选，用于减少抖动）
            biomechanical_constraints: 生物力学约束（可选，用于验证测量值）
        """
        self.transformer = transformer
        self.hand_transformer = hand_transformer
        self.depth_processor = depth_processor
        self.keypoint_stabilizer = keypoint_stabilizer
        self.biomechanical_constraints = biomechanical_constraints
        
        # 人体分割器
        seg_config = SegmentationConfig(
            contour_expand_pixels=15,
            depth_tolerance=0.3,
            enable_depth_filter=True
        )
        self.body_segmentation = BodySegmentation(seg_config)
        
        # 配置参数
        self.min_visibility = _env_float("MEAS_MIN_VISIBILITY", 0.2)
        self.linear_scale = _env_float("MEAS_LINEAR_SCALE", 1.0)

        # 深度处理开关
        self.enable_depth_enhancement = _env_bool("MEAS_DEPTH_ENHANCEMENT", True)
        # 关键点稳定开关
        self.enable_keypoint_stabilization = _env_bool("MEAS_KEYPOINT_STABILIZATION", True)
        # 生物力学约束开关
        self.enable_biomechanical_constraints = _env_bool("MEAS_BIOMECHANICAL_CONSTRAINTS", True)
        # 人体分割开关
        self.enable_body_segmentation = _env_bool("MEAS_BODY_SEGMENTATION", True)
        
        # 最近一次约束检查结果
        self.last_constraints_result: Optional['ConstraintsResult'] = None
        # 最近一次人体掩码
        self.last_body_mask: Optional[np.ndarray] = None
    
    def _preprocess_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """
        预处理深度图
        
        如果启用了深度处理器，进行深度图增强；
        否则返回原始深度图。
        
        Args:
            depth_image: 原始深度图
        
        Returns:
            处理后的深度图
        """
        if not self.enable_depth_enhancement:
            return depth_image
        
        if self.depth_processor is None:
            return depth_image
        
        try:
            # 使用深度处理器进行增强
            return self.depth_processor.process(
                depth_image,
                depth_scale=self.transformer.depth_scale
            )
        except Exception:
            # 处理失败时返回原始深度图
            return depth_image
    
    def reset(self):
        """重置引擎状态"""
        if self.depth_processor is not None:
            self.depth_processor.reset()
        if self.keypoint_stabilizer is not None:
            self.keypoint_stabilizer.reset()
        if self.body_segmentation is not None:
            self.body_segmentation.reset()
        self.last_constraints_result = None
        self.last_body_mask = None
    
    def _stabilize_3d_points(
        self,
        points_3d: list,
        timestamp: float
    ) -> list:
        """
        稳定 3D 关键点
        
        Args:
            points_3d: 3D 点列表
            timestamp: 时间戳
        
        Returns:
            稳定后的 3D 点列表
        """
        if not self.enable_keypoint_stabilization:
            return points_3d
        
        if self.keypoint_stabilizer is None:
            return points_3d
        
        try:
            return self.keypoint_stabilizer.stabilize_points_3d(points_3d, timestamp)
        except Exception:
            return points_3d
    
    def calculate_measurements(self, result: Any, depth_image: np.ndarray,
                               image_width: int, image_height: int,
                               timestamp: Optional[float] = None) -> Optional[Dict[str, Optional[float]]]:
        """
        计算测量值
        
        Args:
            result: HolisticResult 检测结果
            depth_image: 深度图像
            image_width: 图像宽度
            image_height: 图像高度
            timestamp: 时间戳（用于关键点稳定）
        
        Returns:
            测量值字典（单位：cm），或 None
        """
        if not result.pose.detected:
            return None
        
        if timestamp is None:
            timestamp = time.time()
        
        # 深度图预处理（如果启用了深度处理器）
        processed_depth = self._preprocess_depth(depth_image)
        
        # 人体分割：生成掩码并过滤背景深度
        body_mask = None
        if self.enable_body_segmentation:
            # 先获取躯干参考深度
            reference_depth = self.transformer.get_torso_reference_depth(
                result.pose, depth_image, (image_width, image_height)
            )
            # 生成人体掩码
            body_mask = self.body_segmentation.create_body_mask(
                result.pose,
                (image_width, image_height),
                reference_depth=reference_depth,
                depth_frame=processed_depth,
                depth_scale=self.transformer.depth_scale
            )
            self.last_body_mask = body_mask
            
            # 应用掩码到深度图（背景设为0）
            if body_mask is not None:
                processed_depth = self.body_segmentation.get_masked_depth(
                    processed_depth, body_mask
                )
        
        # 2D → 3D 坐标变换
        points_3d = self.transformer.transform_with_filter(
            result.pose,
            processed_depth,
            image_size=(image_width, image_height),
            min_visibility=self.min_visibility,
            depth_range=(0.3, 4.0),
            use_enhanced=True
        )
        
        # 3D 关键点稳定（如果启用了稳定器）
        points_3d = self._stabilize_3d_points(points_3d, timestamp)
        
        # 处理左手 - 直接用手部手腕替换身体手腕，形成肘部到手腕的小臂骨骼
        if result.left_hand and result.left_hand.detected and self.hand_transformer is not None:
            # 获取身体肘部深度作为参考
            try:
                elbow_lm = result.pose.landmarks[13]  # 左肘
                elbow_u = int(elbow_lm.x * image_width)
                elbow_v = int(elbow_lm.y * image_height)
                left_elbow_depth = self.transformer.apply_median_filter(processed_depth, elbow_u, elbow_v)
            except Exception:
                left_elbow_depth = None
            
            left_hand_3d = self.hand_transformer.hand_landmarks_to_3d(
                result.left_hand,
                processed_depth,
                (image_width, image_height),
                body_wrist_depth=left_elbow_depth  # 用肘部深度作为参考
            )
            
            if len(left_hand_3d) == 21 and len(points_3d) >= 33:
                # 关键：用手部手腕(0)直接替换身体手腕(15)
                # 这样小臂骨骼 = 肘部(13) -> 手腕(15) = 肘部 -> 手部手腕
                if left_hand_3d[0].is_valid():
                    points_3d[15] = left_hand_3d[0]  # 身体左手腕 <- 手部手腕
                
                # 手指点映射
                points_3d[19] = left_hand_3d[5]   # 左食指 <- 手部食指MCP
                points_3d[17] = left_hand_3d[17]  # 左小指 <- 手部小指MCP
                points_3d[21] = left_hand_3d[1]   # 左拇指 <- 手部拇指CMC
        
        # 处理右手 - 直接用手部手腕替换身体手腕，形成肘部到手腕的小臂骨骼
        if result.right_hand and result.right_hand.detected and self.hand_transformer is not None:
            # 获取身体肘部深度作为参考
            try:
                elbow_lm = result.pose.landmarks[14]  # 右肘
                elbow_u = int(elbow_lm.x * image_width)
                elbow_v = int(elbow_lm.y * image_height)
                right_elbow_depth = self.transformer.apply_median_filter(processed_depth, elbow_u, elbow_v)
            except Exception:
                right_elbow_depth = None
            
            right_hand_3d = self.hand_transformer.hand_landmarks_to_3d(
                result.right_hand,
                processed_depth,
                (image_width, image_height),
                body_wrist_depth=right_elbow_depth  # 用肘部深度作为参考
            )
            
            if len(right_hand_3d) == 21 and len(points_3d) >= 33:
                # 关键：用手部手腕(0)直接替换身体手腕(16)
                # 这样小臂骨骼 = 肘部(14) -> 手腕(16) = 肘部 -> 手部手腕
                if right_hand_3d[0].is_valid():
                    points_3d[16] = right_hand_3d[0]  # 身体右手腕 <- 手部手腕
                
                # 手指点映射
                points_3d[20] = right_hand_3d[5]   # 右食指 <- 手部食指MCP
                points_3d[18] = right_hand_3d[17]  # 右小指 <- 手部小指MCP
                points_3d[22] = right_hand_3d[1]   # 右拇指 <- 手部拇指CMC
        
        # 构建 3D 骨骼
        skeleton_3d = Skeleton3D.from_points(points_3d)
        
        # 延迟导入避免循环依赖
        from src.measurement.linear_measurements import LinearMeasurements
        
        # 计算骨骼长度
        bones = LinearMeasurements.calculate_bike_fitting_bones(skeleton_3d)
        height_m = LinearMeasurements.calculate_height(skeleton_3d)
        shoulder_width_m = LinearMeasurements.calculate_shoulder_width(skeleton_3d)
        arm_span_m = LinearMeasurements.calculate_arm_span(skeleton_3d)
        leg_length_m = LinearMeasurements.calculate_leg_length(skeleton_3d)
        sitting_height_m = LinearMeasurements.calculate_sitting_height(skeleton_3d)
        pelvic_width_m = LinearMeasurements.calculate_pelvic_width(skeleton_3d)
        upper_limb_m = LinearMeasurements.calculate_upper_limb_length(skeleton_3d)
        lower_limb_m = LinearMeasurements.calculate_lower_limb_length(skeleton_3d)
        trunk_m = LinearMeasurements.calculate_trunk_length(skeleton_3d)
        foot_length_m = LinearMeasurements.calculate_foot_length(skeleton_3d)

        # 手长：从手部关键点计算（手腕→中指MCP），或从臂长估算
        left_hand_length_m = 0.0
        right_hand_length_m = 0.0
        if result.left_hand and result.left_hand.detected and len(result.left_hand.landmarks_3d) == 21:
            wrist = result.left_hand.landmarks_3d[0]
            mcp = result.left_hand.landmarks_3d[9]
            if wrist and mcp:
                left_hand_length_m = ((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2 + (mcp.z - wrist.z)**2) ** 0.5
        if result.right_hand and result.right_hand.detected and len(result.right_hand.landmarks_3d) == 21:
            wrist = result.right_hand.landmarks_3d[0]
            mcp = result.right_hand.landmarks_3d[9]
            if wrist and mcp:
                right_hand_length_m = ((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2 + (mcp.z - wrist.z)**2) ** 0.5
        # 手部检测不可用时，从前臂长度估算手长（人体比例：手长≈前臂长×0.9）
        if left_hand_length_m <= 0 and right_hand_length_m <= 0:
            forearm = bones.get('left_elbow_to_left_wrist') or bones.get('right_elbow_to_right_wrist') or 0.0
            if forearm > 0:
                left_hand_length_m = forearm * 0.9
        hand_length_m = 0.0
        if left_hand_length_m > 0 and right_hand_length_m > 0:
            hand_length_m = (left_hand_length_m + right_hand_length_m) / 2
        else:
            hand_length_m = left_hand_length_m or right_hand_length_m

        def _bone_cm_or_none(*bone_keys: str) -> Optional[float]:
            for bk in bone_keys:
                if not bk:
                    continue
                v = bones.get(bk, None)
                vv = to_cm_or_none(v, scale=self.linear_scale)
                if vv is not None:
                    return vv
            return None

        measurements = {
            '身高': to_cm_or_none(height_m, scale=self.linear_scale),
            '肩宽': to_cm_or_none(shoulder_width_m, scale=self.linear_scale),
            '臂展': to_cm_or_none(arm_span_m, scale=self.linear_scale),
            '腿长': to_cm_or_none(leg_length_m, scale=self.linear_scale),
            '坐高': to_cm_or_none(sitting_height_m, scale=self.linear_scale),
            '骨盆宽': to_cm_or_none(pelvic_width_m, scale=self.linear_scale),
            '上肢长': to_cm_or_none(upper_limb_m, scale=self.linear_scale),
            '下肢长': to_cm_or_none(lower_limb_m, scale=self.linear_scale),
            '颈臀长': to_cm_or_none(trunk_m, scale=self.linear_scale),
            '手长': to_cm_or_none(hand_length_m, scale=self.linear_scale),
            '足长': to_cm_or_none(foot_length_m, scale=self.linear_scale),
            '脊柱底部到中部': _bone_cm_or_none('spine_base_to_spine_mid'),
            '脊柱中部到肩部': _bone_cm_or_none('spine_mid_to_spine_shoulder'),
            '脊柱肩部到头部': _bone_cm_or_none('spine_shoulder_to_head'),
            '脊柱肩部到左肩': _bone_cm_or_none('spine_shoulder_to_shoulder_left', 'spine_shoulder_to_left_shoulder'),
            '左肩到左肘': _bone_cm_or_none('shoulder_left_to_elbow_left', 'left_shoulder_to_left_elbow'),
            '左肘到左腕': _bone_cm_or_none('elbow_left_to_wrist_left', 'left_elbow_to_left_wrist'),
            '左腕到左手': _bone_cm_or_none('wrist_left_to_hand_left', 'left_wrist_to_left_hand'),
            '脊柱肩部到右肩': _bone_cm_or_none('spine_shoulder_to_shoulder_right', 'spine_shoulder_to_right_shoulder'),
            '右肩到右肘': _bone_cm_or_none('shoulder_right_to_elbow_right', 'right_shoulder_to_right_elbow'),
            '右肘到右腕': _bone_cm_or_none('elbow_right_to_wrist_right', 'right_elbow_to_right_wrist'),
            '右腕到右手': _bone_cm_or_none('wrist_right_to_hand_right', 'right_wrist_to_right_hand'),
            '脊柱底部到左髋': _bone_cm_or_none('spine_base_to_hip_left', 'spine_base_to_left_hip'),
            '左髋到左膝': _bone_cm_or_none('hip_left_to_knee_left', 'left_hip_to_left_knee'),
            '左膝到左踝': _bone_cm_or_none('knee_left_to_ankle_left', 'left_knee_to_left_ankle'),
            '左踝到左脚': _bone_cm_or_none('ankle_left_to_foot_left', 'left_ankle_to_left_foot'),
            '脊柱底部到右髋': _bone_cm_or_none('spine_base_to_hip_right', 'spine_base_to_right_hip'),
            '右髋到右膝': _bone_cm_or_none('hip_right_to_knee_right', 'right_hip_to_right_knee'),
            '右膝到右踝': _bone_cm_or_none('knee_right_to_ankle_right', 'right_knee_to_right_ankle'),
            '右踝到右脚': _bone_cm_or_none('ankle_right_to_foot_right', 'right_ankle_to_right_foot'),
        }

        # 生物力学约束验证（如果启用）
        if self.enable_biomechanical_constraints and self.biomechanical_constraints is not None:
            try:
                self.last_constraints_result = self.biomechanical_constraints.validate(measurements)
            except Exception:
                self.last_constraints_result = None

        return measurements

    def _calculate_from_pixel_landmarks(
        self, result: Any, img_w: int, img_h: int,
        intrinsics: Any, timestamp: float
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        使用 2D 像素关键点 + 身体比例法计算测量值

        核心原理：身高与头肩距的比值在不同距离下恒定
            real_height = height_px / ref_px * ref_real_cm

        水平测量用针孔模型 + 距离估算：
            distance_est = fx * ref_real / ref_px
            horizontal_cm = px * distance_est / fx

        Args:
            result: HolisticResult 检测结果
            img_w, img_h: 图像尺寸（像素）
            intrinsics: 相机内参（含 fx, fy）
            timestamp: 时间戳
        Returns:
            测量值字典（单位：cm），或 None
        """
        lm = result.pose.landmarks
        min_vis = min(self.min_visibility, 0.2)

        def _px(idx: int) -> Optional[Tuple[float, float]]:
            """获取关键点的像素坐标，不可见则返回 None"""
            if idx >= len(lm):
                return None
            p = lm[idx]
            if p.visibility < min_vis:
                return None
            return (p.x * img_w, p.y * img_h)

        # 常用关键点索引
        NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
        L_HIP, R_HIP = 23, 24
        L_KNEE, R_KNEE = 25, 26
        L_ANKLE, R_ANKLE = 27, 28
        L_ELBOW, R_ELBOW = 13, 14
        L_WRIST, R_WRIST = 15, 16
        L_PINKY, R_PINKY = 17, 18
        L_INDEX, R_INDEX = 19, 20
        L_HEEL, R_HEEL = 29, 30
        L_FOOT_INDEX, R_FOOT_INDEX = 31, 32

        def _mid_px(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

        def _y_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return abs(a[1] - b[1])

        def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

        # --- 参照物：头肩距（鼻→肩中心 Y 轴距离，像素）---
        nose_px = _px(NOSE)
        ls_px = _px(L_SHOULDER)
        rs_px = _px(R_SHOULDER)
        if not nose_px or not (ls_px or rs_px):
            return None

        if ls_px and rs_px:
            sc_px = _mid_px(ls_px, rs_px)
        else:
            sc_px = ls_px or rs_px

        ref_px = _y_dist(nose_px, sc_px)
        # 头肩距（鼻→肩中心）≈ 24cm（成人平均值）
        REF_REAL_CM = 24.0

        if ref_px < 5:
            return None

        # --- 身高（Y 轴像素距离累加 + 比例转换）---
        lh_px = _px(L_HIP)
        rh_px = _px(R_HIP)
        lk_px = _px(L_KNEE)
        rk_px = _px(R_KNEE)
        la_px = _px(L_ANKLE)
        ra_px = _px(R_ANKLE)

        if lh_px and rh_px:
            hc_px = _mid_px(lh_px, rh_px)
        else:
            hc_px = lh_px or rh_px

        height_px = 0.0
        has_lower = False

        if hc_px:
            height_px += _y_dist(nose_px, sc_px)     # 颈段
            height_px += _y_dist(sc_px, hc_px)       # 躯干

            if (lk_px or rk_px) and (la_px or ra_px):
                k_px = _mid_px(lk_px, rk_px) if (lk_px and rk_px) else (lk_px or rk_px)
                a_px = _mid_px(la_px, ra_px) if (la_px and ra_px) else (la_px or ra_px)
                if k_px and a_px:
                    height_px += _y_dist(hc_px, k_px)  # 大腿
                    height_px += _y_dist(k_px, a_px)   # 小腿
                    has_lower = True

        if height_px < ref_px:
            return None

        # 身高 = height_px / ref_px * REF_REAL_CM（比值法，与距离无关）
        if has_lower:
            height_cm = (height_px / ref_px) * REF_REAL_CM
        else:
            # 仅有上半身时，从上半身比例估算全长
            # 上半身（鼻→髋）约占身高 48%
            upper_body_cm = (height_px / ref_px) * REF_REAL_CM
            height_cm = upper_body_cm / 0.48

        # --- 水平测量（针孔模型 + 距离估算）---
        # 从参照物估算距离：D = fx * ref_real / ref_px
        # 然后：horizontal_cm = px * D / fx = px * ref_real / ref_px
        fx = intrinsics.fx if intrinsics and hasattr(intrinsics, 'fx') and intrinsics.fx > 0 else (img_w * 1.2)

        def _h_measure_cm(px_dist: float) -> float:
            """将像素水平距离转换为厘米"""
            return (px_dist * REF_REAL_CM) / ref_px

        # --- 肩宽 ---
        shoulder_cm = 0.0
        if ls_px and rs_px:
            shoulder_cm = _h_measure_cm(_euclid(ls_px, rs_px))

        # --- 臂展 ---
        arm_span_cm = 0.0
        lw_px = _px(L_WRIST)
        rw_px = _px(R_WRIST)
        if lw_px and rw_px:
            arm_span_cm = _h_measure_cm(_euclid(lw_px, rw_px))

        # --- 腿长 ---
        leg_cm = 0.0
        if hc_px and (la_px or ra_px):
            a_px = _mid_px(la_px, ra_px) if (la_px and ra_px) else (la_px or ra_px)
            if a_px:
                leg_cm = _h_measure_cm(_euclid(hc_px, a_px))

        # --- 坐高 ---
        sitting_cm = 0.0
        if hc_px:
            sitting_cm = (height_cm * 0.48) if not has_lower else (height_cm * 0.52)

        # --- 骨盆宽 ---
        pelvic_cm = 0.0
        if lh_px and rh_px:
            pelvic_cm = _h_measure_cm(_euclid(lh_px, rh_px))

        # --- 上肢长 ---
        upper_limb_cm = 0.0
        le_px = _px(L_ELBOW)
        re_px = _px(R_ELBOW)
        if ls_px and le_px and lw_px:
            upper_limb_cm = _h_measure_cm(_euclid(ls_px, le_px) + _euclid(le_px, lw_px))
        elif rs_px and re_px and rw_px:
            upper_limb_cm = _h_measure_cm(_euclid(rs_px, re_px) + _euclid(re_px, rw_px))

        # --- 下肢长 ---
        lower_limb_cm = 0.0
        if lh_px and la_px:
            lower_limb_cm = _h_measure_cm(_euclid(lh_px, la_px))
        elif rh_px and ra_px:
            lower_limb_cm = _h_measure_cm(_euclid(rh_px, ra_px))

        # --- 颈臀长 ---
        trunk_cm = 0.0
        if hc_px:
            trunk_cm = _h_measure_cm(_y_dist(nose_px, hc_px))

        # --- 足长 ---
        foot_cm = 0.0
        l_heel_px = _px(L_HEEL)
        r_heel_px = _px(R_HEEL)
        l_fi_px = _px(L_FOOT_INDEX)
        r_fi_px = _px(R_FOOT_INDEX)
        if l_heel_px and l_fi_px:
            foot_cm = _h_measure_cm(_euclid(l_heel_px, l_fi_px))
        elif r_heel_px and r_fi_px:
            foot_cm = _h_measure_cm(_euclid(r_heel_px, r_fi_px))
        elif la_px and lk_px:
            # 从小腿长估算足长（足长 ≈ 小腿长 × 0.55）
            calf_px = _euclid(lk_px, la_px) if (lk_px and la_px) else 0
            if calf_px > 0:
                foot_cm = _h_measure_cm(calf_px) * 0.55

        # 手长估算
        hand_cm = 0.0
        if le_px and lw_px:
            forearm_cm = _h_measure_cm(_euclid(le_px, lw_px))
            hand_cm = forearm_cm * 0.9
        elif re_px and rw_px:
            forearm_cm = _h_measure_cm(_euclid(re_px, rw_px))
            hand_cm = forearm_cm * 0.9

        # 应用 linear_scale
        s = self.linear_scale
        measurements = {
            '身高': round(height_cm * s, 1) if height_cm > 0 else None,
            '肩宽': round(shoulder_cm * s, 1) if shoulder_cm > 0 else None,
            '臂展': round(arm_span_cm * s, 1) if arm_span_cm > 0 else None,
            '腿长': round(leg_cm * s, 1) if leg_cm > 0 else None,
            '坐高': round(sitting_cm * s, 1) if sitting_cm > 0 else None,
            '骨盆宽': round(pelvic_cm * s, 1) if pelvic_cm > 0 else None,
            '上肢长': round(upper_limb_cm * s, 1) if upper_limb_cm > 0 else None,
            '下肢长': round(lower_limb_cm * s, 1) if lower_limb_cm > 0 else None,
            '颈臀长': round(trunk_cm * s, 1) if trunk_cm > 0 else None,
            '手长': round(hand_cm * s, 1) if hand_cm > 0 else None,
            '足长': round(foot_cm * s, 1) if foot_cm > 0 else None,
        }

        logger.debug(f"像素比例法: ref_px={ref_px:.1f}, height_px={height_px:.1f}, "
                     f"height_cm={height_cm:.1f}, shoulder_cm={shoulder_cm:.1f}")

        if self.enable_biomechanical_constraints and self.biomechanical_constraints is not None:
            try:
                self.last_constraints_result = self.biomechanical_constraints.validate(measurements)
            except Exception:
                self.last_constraints_result = None

        return measurements

    def calculate_measurements_from_world_landmarks(
        self, result: Any, timestamp: Optional[float] = None,
        image_width: int = 0, image_height: int = 0,
        intrinsics: Any = None
    ) -> Optional[Dict[str, Optional[float]]]:
        """
        使用 MediaPipe 2D 像素关键点计算测量值（无需深度图）

        适用于 OpenCV 模式（无 RealSense 深度相机）。
        使用 2D 像素坐标 + 身体比例法（头肩距参照物）将像素距离转换为厘米。
        此方法不依赖 world_landmarks 的绝对米制坐标（它们不可靠）。

        原理：
        - 身高与头肩距的比值是恒定的（与距离无关）
        - real_height = height_px / ref_px * ref_real_cm
        - 水平测量用针孔模型：cm = px * D / f，D 从参照物推算

        Args:
            result: HolisticResult 检测结果
            timestamp: 时间戳
            image_width: 图像宽度（像素）
            image_height: 图像高度（像素）
            intrinsics: 相机内参（可选，用于水平测量的距离估算）

        Returns:
            测量值字典（单位：cm），或 None
        """
        if not result.pose.detected:
            return None

        if timestamp is None:
            timestamp = time.time()

        # 优先用 2D 像素坐标 + 身体比例法
        if image_width > 0 and image_height > 0 and result.pose.landmarks and len(result.pose.landmarks) >= 33:
            return self._calculate_from_pixel_landmarks(
                result, image_width, image_height, intrinsics, timestamp
            )

        # 回退到 world_landmarks（精度差，但总比没有好）
        if not result.pose.world_landmarks or len(result.pose.world_landmarks) < 33:
            return None

        wl = result.pose.world_landmarks

        wl_min_vis = min(self.min_visibility, 0.2)
        points_3d: list = [None] * 33
        for i in range(min(33, len(wl))):
            lm = wl[i]
            if lm.visibility >= wl_min_vis:
                points_3d[i] = Point3D(x=lm.x, y=lm.y, z=lm.z, confidence=lm.visibility)

        skeleton_3d = Skeleton3D.from_points(points_3d)

        from src.measurement.linear_measurements import LinearMeasurements

        bones = LinearMeasurements.calculate_bike_fitting_bones(skeleton_3d)
        height_m = LinearMeasurements.calculate_height(skeleton_3d)
        shoulder_width_m = LinearMeasurements.calculate_shoulder_width(skeleton_3d)
        arm_span_m = LinearMeasurements.calculate_arm_span(skeleton_3d)
        leg_length_m = LinearMeasurements.calculate_leg_length(skeleton_3d)
        sitting_height_m = LinearMeasurements.calculate_sitting_height(skeleton_3d)
        pelvic_width_m = LinearMeasurements.calculate_pelvic_width(skeleton_3d)
        upper_limb_m = LinearMeasurements.calculate_upper_limb_length(skeleton_3d)
        lower_limb_m = LinearMeasurements.calculate_lower_limb_length(skeleton_3d)
        trunk_m = LinearMeasurements.calculate_trunk_length(skeleton_3d)
        foot_length_m = LinearMeasurements.calculate_foot_length(skeleton_3d)

        # 手长：从手部关键点计算，或从臂长估算
        left_hand_length_m = 0.0
        right_hand_length_m = 0.0
        if result.left_hand and result.left_hand.detected and len(result.left_hand.landmarks_3d) == 21:
            wrist = result.left_hand.landmarks_3d[0]
            mcp = result.left_hand.landmarks_3d[9]
            if wrist and mcp:
                left_hand_length_m = ((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2 + (mcp.z - wrist.z)**2) ** 0.5
        if result.right_hand and result.right_hand.detected and len(result.right_hand.landmarks_3d) == 21:
            wrist = result.right_hand.landmarks_3d[0]
            mcp = result.right_hand.landmarks_3d[9]
            if wrist and mcp:
                right_hand_length_m = ((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2 + (mcp.z - wrist.z)**2) ** 0.5
        # 手部检测不可用时，从前臂长度估算手长
        if left_hand_length_m <= 0 and right_hand_length_m <= 0:
            forearm = bones.get('left_elbow_to_left_wrist') or bones.get('right_elbow_to_right_wrist') or 0.0
            if forearm > 0:
                left_hand_length_m = forearm * 0.9
        hand_length_m = 0.0
        if left_hand_length_m > 0 and right_hand_length_m > 0:
            hand_length_m = (left_hand_length_m + right_hand_length_m) / 2
        else:
            hand_length_m = left_hand_length_m or right_hand_length_m

        def _bone_cm_or_none(*bone_keys: str) -> Optional[float]:
            for bk in bone_keys:
                if not bk:
                    continue
                v = bones.get(bk, None)
                vv = to_cm_or_none(v, scale=self.linear_scale)
                if vv is not None:
                    return vv
            return None

        measurements = {
            '身高': to_cm_or_none(height_m, scale=self.linear_scale),
            '肩宽': to_cm_or_none(shoulder_width_m, scale=self.linear_scale),
            '臂展': to_cm_or_none(arm_span_m, scale=self.linear_scale),
            '腿长': to_cm_or_none(leg_length_m, scale=self.linear_scale),
            '坐高': to_cm_or_none(sitting_height_m, scale=self.linear_scale),
            '骨盆宽': to_cm_or_none(pelvic_width_m, scale=self.linear_scale),
            '上肢长': to_cm_or_none(upper_limb_m, scale=self.linear_scale),
            '下肢长': to_cm_or_none(lower_limb_m, scale=self.linear_scale),
            '颈臀长': to_cm_or_none(trunk_m, scale=self.linear_scale),
            '手长': to_cm_or_none(hand_length_m, scale=self.linear_scale),
            '足长': to_cm_or_none(foot_length_m, scale=self.linear_scale),
            '脊柱底部到中部': _bone_cm_or_none('spine_base_to_spine_mid'),
            '脊柱中部到肩部': _bone_cm_or_none('spine_mid_to_spine_shoulder'),
            '脊柱肩部到头部': _bone_cm_or_none('spine_shoulder_to_head'),
            '脊柱肩部到左肩': _bone_cm_or_none('spine_shoulder_to_shoulder_left', 'spine_shoulder_to_left_shoulder'),
            '左肩到左肘': _bone_cm_or_none('shoulder_left_to_elbow_left', 'left_shoulder_to_left_elbow'),
            '左肘到左腕': _bone_cm_or_none('elbow_left_to_wrist_left', 'left_elbow_to_left_wrist'),
            '左腕到左手': _bone_cm_or_none('wrist_left_to_hand_left', 'left_wrist_to_left_hand'),
            '脊柱肩部到右肩': _bone_cm_or_none('spine_shoulder_to_shoulder_right', 'spine_shoulder_to_right_shoulder'),
            '右肩到右肘': _bone_cm_or_none('shoulder_right_to_elbow_right', 'right_shoulder_to_right_elbow'),
            '右肘到右腕': _bone_cm_or_none('elbow_right_to_wrist_right', 'right_elbow_to_right_wrist'),
            '右腕到右手': _bone_cm_or_none('wrist_right_to_hand_right', 'right_wrist_to_right_hand'),
            '脊柱底部到左髋': _bone_cm_or_none('spine_base_to_hip_left', 'spine_base_to_left_hip'),
            '左髋到左膝': _bone_cm_or_none('hip_left_to_knee_left', 'left_hip_to_left_knee'),
            '左膝到左踝': _bone_cm_or_none('knee_left_to_ankle_left', 'left_knee_to_left_ankle'),
            '左踝到左脚': _bone_cm_or_none('ankle_left_to_foot_left', 'left_ankle_to_left_foot'),
            '脊柱底部到右髋': _bone_cm_or_none('spine_base_to_hip_right', 'spine_base_to_right_hip'),
            '右髋到右膝': _bone_cm_or_none('hip_right_to_knee_right', 'right_hip_to_right_knee'),
            '右膝到右踝': _bone_cm_or_none('knee_right_to_ankle_right', 'right_knee_to_right_ankle'),
            '右踝到右脚': _bone_cm_or_none('ankle_right_to_foot_right', 'right_ankle_to_right_foot'),
        }

        if self.enable_biomechanical_constraints and self.biomechanical_constraints is not None:
            try:
                self.last_constraints_result = self.biomechanical_constraints.validate(measurements)
            except Exception:
                self.last_constraints_result = None

        return measurements
