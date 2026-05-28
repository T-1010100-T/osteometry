"""
ROI优化器属性测试

**Feature: holistic-hand-integration, Property 14: ROI计算正确性**
**Validates: Requirements 8.3, 8.4**
"""
import sys
from pathlib import Path

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.pose_estimator import PoseResult, Landmark
from src.core.roi_optimizer import ROIOptimizer, ROI
from src.core.constants import PoseLandmark


def create_pose_result_with_wrist(
    left_wrist_x: float = 0.3,
    left_wrist_y: float = 0.5,
    right_wrist_x: float = 0.7,
    right_wrist_y: float = 0.5,
    visibility: float = 0.9
) -> PoseResult:
    """创建包含手腕位置的姿态结果"""
    landmarks = []
    
    for i in range(33):
        if i == PoseLandmark.LEFT_WRIST:
            landmarks.append(Landmark(x=left_wrist_x, y=left_wrist_y, z=0, visibility=visibility))
        elif i == PoseLandmark.RIGHT_WRIST:
            landmarks.append(Landmark(x=right_wrist_x, y=right_wrist_y, z=0, visibility=visibility))
        elif i == PoseLandmark.LEFT_ELBOW:
            landmarks.append(Landmark(x=left_wrist_x - 0.1, y=left_wrist_y - 0.1, z=0, visibility=visibility))
        elif i == PoseLandmark.RIGHT_ELBOW:
            landmarks.append(Landmark(x=right_wrist_x + 0.1, y=right_wrist_y - 0.1, z=0, visibility=visibility))
        else:
            landmarks.append(Landmark(x=0.5, y=0.5, z=0, visibility=0.5))
    
    return PoseResult(landmarks=landmarks, detected=True)


class TestROICalculation:
    """
    ROI计算正确性测试
    
    **Property 14: ROI计算正确性**
    """
    
    def test_roi_centered_on_wrist(self):
        """
        **Property 14: ROI计算正确性**
        ROI应以手腕为中心（考虑偏移）
        """
        optimizer = ROIOptimizer(roi_size=300)
        pose = create_pose_result_with_wrist(left_wrist_x=0.5, left_wrist_y=0.5)
        image_size = (640, 480)
        
        roi = optimizer.get_hand_roi(pose, image_size, "left")
        
        assert roi is not None
        
        # 手腕像素位置
        wrist_px = int(0.5 * 640)
        wrist_py = int(0.5 * 480)
        
        # ROI中心应接近手腕位置（允许偏移）
        roi_center_x = roi.x + roi.width // 2
        roi_center_y = roi.y + roi.height // 2
        
        # 允许一定偏移（因为ROI会向手指方向偏移）
        assert abs(roi_center_x - wrist_px) < 100
        assert abs(roi_center_y - wrist_py) < 100
    
    def test_roi_size_is_correct(self):
        """
        **Property 14: ROI计算正确性**
        ROI尺寸应为指定大小
        """
        roi_size = 300
        optimizer = ROIOptimizer(roi_size=roi_size)
        pose = create_pose_result_with_wrist()
        image_size = (640, 480)
        
        roi = optimizer.get_hand_roi(pose, image_size, "left")
        
        assert roi is not None
        assert roi.width == roi_size
        assert roi.height == roi_size
    
    def test_roi_clipped_to_image_boundary(self):
        """
        **Property 14: ROI计算正确性**
        ROI应裁剪到图像边界
        """
        optimizer = ROIOptimizer(roi_size=300)
        # 手腕在图像边缘
        pose = create_pose_result_with_wrist(left_wrist_x=0.05, left_wrist_y=0.05)
        image_size = (640, 480)
        
        roi = optimizer.get_hand_roi(pose, image_size, "left")
        
        assert roi is not None
        assert roi.x >= 0
        assert roi.y >= 0
        assert roi.x + roi.width <= image_size[0]
        assert roi.y + roi.height <= image_size[1]
    
    def test_roi_at_right_edge(self):
        """ROI在右边缘时应正确裁剪"""
        optimizer = ROIOptimizer(roi_size=300)
        pose = create_pose_result_with_wrist(right_wrist_x=0.95, right_wrist_y=0.5)
        image_size = (640, 480)
        
        roi = optimizer.get_hand_roi(pose, image_size, "right")
        
        assert roi is not None
        assert roi.x + roi.width <= image_size[0]
    
    def test_roi_at_bottom_edge(self):
        """ROI在底部边缘时应正确裁剪"""
        optimizer = ROIOptimizer(roi_size=300)
        pose = create_pose_result_with_wrist(left_wrist_x=0.5, left_wrist_y=0.95)
        image_size = (640, 480)
        
        roi = optimizer.get_hand_roi(pose, image_size, "left")
        
        assert roi is not None
        assert roi.y + roi.height <= image_size[1]
    
    @given(
        wrist_x=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
        wrist_y=st.floats(min_value=0.1, max_value=0.9, allow_nan=False)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_roi_always_within_image(self, wrist_x: float, wrist_y: float):
        """
        **Property 14: ROI计算正确性**
        任何手腕位置的ROI都应在图像内
        """
        optimizer = ROIOptimizer(roi_size=300)
        pose = create_pose_result_with_wrist(left_wrist_x=wrist_x, left_wrist_y=wrist_y)
        image_size = (640, 480)
        
        roi = optimizer.get_hand_roi(pose, image_size, "left")
        
        assert roi is not None
        assert 0 <= roi.x < image_size[0]
        assert 0 <= roi.y < image_size[1]
        assert roi.x + roi.width <= image_size[0]
        assert roi.y + roi.height <= image_size[1]


class TestROICropping:
    """ROI裁剪测试"""
    
    def test_crop_roi_returns_correct_size(self):
        """crop_roi 应返回正确尺寸的图像"""
        optimizer = ROIOptimizer(roi_size=300)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = ROI(x=100, y=100, width=300, height=300)
        
        cropped = optimizer.crop_roi(image, roi)
        
        assert cropped.shape == (300, 300, 3)
    
    def test_crop_roi_preserves_content(self):
        """crop_roi 应保留图像内容"""
        optimizer = ROIOptimizer(roi_size=100)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 在ROI区域内设置特定值
        image[100:200, 100:200] = 255
        
        roi = ROI(x=100, y=100, width=100, height=100)
        cropped = optimizer.crop_roi(image, roi)
        
        assert np.all(cropped == 255)


class TestLandmarkTransformation:
    """关键点坐标转换测试"""
    
    def test_transform_landmarks_to_original(self):
        """关键点应正确转换回原图坐标"""
        optimizer = ROIOptimizer(roi_size=300)
        
        # ROI在(100, 100)位置
        roi = ROI(x=100, y=100, width=300, height=300)
        image_size = (640, 480)
        
        # ROI内的归一化坐标 (0.5, 0.5) 应该对应原图的 (250, 250)
        landmarks = [Landmark(x=0.5, y=0.5, z=0, visibility=0.9)]
        
        transformed = optimizer.transform_landmarks_to_original(landmarks, roi, image_size)
        
        # 原图像素坐标: 100 + 0.5*300 = 250
        # 归一化: 250/640 ≈ 0.39, 250/480 ≈ 0.52
        expected_x = (100 + 0.5 * 300) / 640
        expected_y = (100 + 0.5 * 300) / 480
        
        assert abs(transformed[0].x - expected_x) < 0.01
        assert abs(transformed[0].y - expected_y) < 0.01
    
    def test_transform_preserves_visibility(self):
        """转换应保留可见性"""
        optimizer = ROIOptimizer()
        roi = ROI(x=0, y=0, width=300, height=300)
        image_size = (640, 480)
        
        landmarks = [Landmark(x=0.5, y=0.5, z=0.1, visibility=0.85)]
        transformed = optimizer.transform_landmarks_to_original(landmarks, roi, image_size)
        
        assert transformed[0].visibility == 0.85
        assert transformed[0].z == 0.1


class TestROIState:
    """ROI状态测试"""
    
    def test_last_roi_saved(self):
        """应保存上一帧的ROI"""
        optimizer = ROIOptimizer(roi_size=300)
        pose = create_pose_result_with_wrist()
        image_size = (640, 480)
        
        roi = optimizer.get_hand_roi(pose, image_size, "left")
        
        assert optimizer.last_left_roi is not None
        assert optimizer.last_left_roi == roi
    
    def test_use_last_roi_when_wrist_invisible(self):
        """手腕不可见时应使用上一帧ROI"""
        optimizer = ROIOptimizer(roi_size=300)
        image_size = (640, 480)
        
        # 先建立ROI
        pose1 = create_pose_result_with_wrist(visibility=0.9)
        roi1 = optimizer.get_hand_roi(pose1, image_size, "left")
        
        # 手腕不可见
        pose2 = create_pose_result_with_wrist(visibility=0.3)
        roi2 = optimizer.get_hand_roi(pose2, image_size, "left")
        
        assert roi2 == roi1
    
    def test_reset_clears_state(self):
        """reset 应清除状态"""
        optimizer = ROIOptimizer(roi_size=300)
        pose = create_pose_result_with_wrist()
        image_size = (640, 480)
        
        optimizer.get_hand_roi(pose, image_size, "left")
        optimizer.get_hand_roi(pose, image_size, "right")
        
        optimizer.reset()
        
        assert optimizer.last_left_roi is None
        assert optimizer.last_right_roi is None


class TestROIDataclass:
    """ROI数据类测试"""
    
    def test_to_tuple(self):
        """to_tuple 应返回正确的元组"""
        roi = ROI(x=10, y=20, width=100, height=200)
        
        assert roi.to_tuple() == (10, 20, 100, 200)
    
    def test_contains_point_inside(self):
        """contains 应正确检测内部点"""
        roi = ROI(x=100, y=100, width=200, height=200)
        
        assert roi.contains(150, 150)
        assert roi.contains(100, 100)
        assert roi.contains(299, 299)
    
    def test_contains_point_outside(self):
        """contains 应正确检测外部点"""
        roi = ROI(x=100, y=100, width=200, height=200)
        
        assert not roi.contains(50, 150)
        assert not roi.contains(150, 50)
        assert not roi.contains(300, 150)
        assert not roi.contains(150, 300)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
