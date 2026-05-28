"""
错误处理与降级行为属性测试

**Feature: holistic-hand-integration, Property 18: 错误降级行为**
**Validates: Requirements 12.2**
"""
import sys
from pathlib import Path

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.hand_exceptions import (
    HandDetectionError,
    HandModelLoadError,
    HandDepthError,
    HandTrackingLostError,
    GestureRecognitionError,
    HandCoordinateError,
    GPUMemoryError
)


class TestExceptionHierarchy:
    """异常类层次结构测试"""
    
    def test_all_exceptions_inherit_from_base(self):
        """所有手部异常应继承自 HandDetectionError"""
        exceptions = [
            HandModelLoadError,
            HandDepthError,
            HandTrackingLostError,
            GestureRecognitionError,
            HandCoordinateError,
            GPUMemoryError
        ]
        
        for exc_class in exceptions:
            assert issubclass(exc_class, HandDetectionError)
    
    def test_exception_message(self):
        """异常应包含正确的消息"""
        exc = HandDetectionError("测试错误")
        assert str(exc) == "测试错误"
        
        exc_with_details = HandDetectionError("测试错误", {'key': 'value'})
        assert "测试错误" in str(exc_with_details)
        assert "key" in str(exc_with_details)
    
    def test_model_load_error(self):
        """HandModelLoadError 应包含模型路径"""
        exc = HandModelLoadError("加载失败", model_path="/path/to/model")
        assert exc.details['model_path'] == "/path/to/model"
    
    def test_depth_error(self):
        """HandDepthError 应包含手部信息"""
        exc = HandDepthError("深度错误", hand="left")
        assert exc.details['hand'] == "left"
    
    def test_tracking_lost_error(self):
        """HandTrackingLostError 应包含丢失帧数"""
        exc = HandTrackingLostError("跟踪丢失", hand="right", lost_frames=30)
        assert exc.details['hand'] == "right"
        assert exc.details['lost_frames'] == 30


class TestErrorGracefulDegradation:
    """
    错误降级行为测试
    
    **Feature: holistic-hand-integration, Property 18: 错误降级行为**
    **Validates: Requirements 12.2**
    """
    
    def test_holistic_estimator_returns_valid_result_on_invalid_input(self):
        """
        **Feature: holistic-hand-integration, Property 18: 错误降级行为**
        **Validates: Requirements 12.2**
        
        无效输入时应返回有效的空结果，不应崩溃
        """
        from src.core.holistic_estimator import HolisticEstimator
        
        estimator = HolisticEstimator(model_complexity=0)
        
        try:
            # 测试无效图像格式
            invalid_image = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D
            result = estimator.detect(invalid_image)
            
            # 应该返回有效的空结果
            assert result is not None
            assert result.pose.detected == False
            assert result.left_hand.detected == False
            assert result.right_hand.detected == False
        finally:
            estimator.close()
    
    def test_holistic_estimator_handles_empty_image(self):
        """
        **Feature: holistic-hand-integration, Property 18: 错误降级行为**
        **Validates: Requirements 12.2**
        
        空图像时应返回有效结果，不应崩溃
        """
        from src.core.holistic_estimator import HolisticEstimator
        
        estimator = HolisticEstimator(model_complexity=0)
        
        try:
            # 全黑图像
            black_image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = estimator.detect(black_image)
            
            # 应该返回有效结果（可能检测不到人）
            assert result is not None
            # 不应该崩溃
        finally:
            estimator.close()
    
    @given(
        width=st.integers(min_value=100, max_value=800),
        height=st.integers(min_value=100, max_value=600)
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_holistic_estimator_handles_various_image_sizes(self, width: int, height: int):
        """
        **Feature: holistic-hand-integration, Property 18: 错误降级行为**
        **Validates: Requirements 12.2**
        
        *For any* valid image size, the estimator SHALL not crash
        """
        from src.core.holistic_estimator import HolisticEstimator
        
        estimator = HolisticEstimator(model_complexity=0)
        
        try:
            # 随机大小的图像
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            result = estimator.detect(image)
            
            # 应该返回有效结果
            assert result is not None
        finally:
            estimator.close()
    
    def test_fallback_mode_property(self):
        """测试降级模式属性"""
        from src.core.holistic_estimator import HolisticEstimator
        
        estimator = HolisticEstimator(model_complexity=0)
        
        try:
            # 正常初始化不应该处于降级模式
            assert hasattr(estimator, '_fallback_mode')
        finally:
            estimator.close()


class TestHandResultEmpty:
    """HandResult 空结果测试"""
    
    def test_empty_hand_result_is_valid(self):
        """空手部结果应该是有效的数据结构"""
        from src.core.hand_result import HandResult
        
        empty_left = HandResult.empty("left")
        empty_right = HandResult.empty("right")
        
        assert empty_left.detected == False
        assert empty_left.handedness == "left"
        assert empty_left.confidence == 0.0
        assert empty_left.landmarks == []
        
        assert empty_right.detected == False
        assert empty_right.handedness == "right"


class TestHolisticResultEmpty:
    """HolisticResult 空结果测试"""
    
    def test_empty_holistic_result_is_valid(self):
        """空 Holistic 结果应该是有效的数据结构"""
        from src.core.hand_result import HolisticResult
        
        result = HolisticResult()
        
        assert result.pose.detected == False
        assert result.left_hand.detected == False
        assert result.right_hand.detected == False
    
    def test_to_pose_result_works_on_empty(self):
        """空结果转换为 PoseResult 应该正常工作"""
        from src.core.hand_result import HolisticResult
        
        result = HolisticResult()
        pose_result = result.to_pose_result()
        
        assert pose_result is not None
        assert pose_result.detected == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
