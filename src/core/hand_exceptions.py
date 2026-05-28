"""
手部检测异常类定义

Requirements: 12.1, 12.2, 12.3, 12.4
"""


class HandDetectionError(Exception):
    """
    手部检测错误基类
    
    所有手部检测相关异常的基类
    """
    
    def __init__(self, message: str = "手部检测错误", details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class HandModelLoadError(HandDetectionError):
    """
    手部模型加载失败
    
    当 MediaPipe Holistic 模型无法加载时抛出
    
    Requirements: 12.1 - 模型加载失败时回退到 Pose-only 模式
    """
    
    def __init__(self, message: str = "手部模型加载失败", model_path: str = None):
        details = {'model_path': model_path} if model_path else {}
        super().__init__(message, details)


class HandDepthError(HandDetectionError):
    """
    手部深度数据错误
    
    当手部区域深度数据无效或不可用时抛出
    
    Requirements: 12.3 - 深度数据不可用时从身体手腕点估算
    """
    
    def __init__(self, message: str = "手部深度数据错误", hand: str = None):
        details = {'hand': hand} if hand else {}
        super().__init__(message, details)


class HandTrackingLostError(HandDetectionError):
    """
    手部跟踪丢失
    
    当手部跟踪连续丢失超过阈值帧数时抛出
    
    Requirements: 12.4 - 跟踪丢失超过30帧时重置状态
    """
    
    def __init__(self, message: str = "手部跟踪丢失", 
                 hand: str = None, lost_frames: int = 0):
        details = {
            'hand': hand,
            'lost_frames': lost_frames
        }
        super().__init__(message, details)


class GestureRecognitionError(HandDetectionError):
    """
    手势识别错误
    
    当手势识别过程中发生错误时抛出
    """
    
    def __init__(self, message: str = "手势识别错误", gesture: str = None):
        details = {'gesture': gesture} if gesture else {}
        super().__init__(message, details)


class HandCoordinateError(HandDetectionError):
    """
    手部坐标转换错误
    
    当手部2D到3D坐标转换失败时抛出
    """
    
    def __init__(self, message: str = "手部坐标转换错误", 
                 landmark_index: int = None):
        details = {'landmark_index': landmark_index} if landmark_index is not None else {}
        super().__init__(message, details)


class GPUMemoryError(HandDetectionError):
    """
    GPU内存不足错误
    
    当GPU内存不足时抛出
    
    Requirements: 12.5 - GPU内存不足时切换到CPU模式
    """
    
    def __init__(self, message: str = "GPU内存不足"):
        super().__init__(message)
