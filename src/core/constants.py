"""
常量定义
MediaPipe Pose 关键点索引和骨骼连接
"""
from enum import IntEnum
from typing import List, Tuple


class PoseLandmark(IntEnum):
    """MediaPipe Pose 33个关键点索引"""
    # 面部
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    
    # 上肢
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    
    # 下肢
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# 关键点名称映射
LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

# 名称到索引的反向映射
LANDMARK_INDICES = {v: k for k, v in LANDMARK_NAMES.items()}


# 骨骼连接定义 (用于绘制和计算)
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    # 面部
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER),
    (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER),
    (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR),
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER),
    (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER),
    (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT),
    
    # 躯干
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    
    # 左臂
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB),
    (PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX),
    
    # 右臂
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB),
    (PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX),
    
    # 左腿
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX),
    
    # 右腿
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
    (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX),
]


# 用于测量的关键关键点
KEY_LANDMARKS = [
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_ANKLE,
]


# 身体测量所需的骨骼段
MEASUREMENT_BONES = {
    # 上肢
    'left_upper_arm': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    'left_forearm': (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    'right_upper_arm': (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    'right_forearm': (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    
    # 下肢
    'left_thigh': (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    'left_calf': (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    'right_thigh': (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    'right_calf': (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    
    # 躯干
    'shoulder_width': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    'hip_width': (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    'left_torso': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    'right_torso': (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
}


# 关节角度测量定义 (三个点定义一个角度)
JOINT_ANGLES = {
    'left_elbow': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    'right_elbow': (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    'left_shoulder': (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    'right_shoulder': (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    'left_hip': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    'right_hip': (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    'left_knee': (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    'right_knee': (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
}


# ============================================================
# 手部关键点定义 (MediaPipe Hand 21点)
# ============================================================

class HandLandmark(IntEnum):
    """MediaPipe Hand 21个关键点索引"""
    WRIST = 0
    
    # 拇指
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    
    # 食指
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    
    # 中指
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    
    # 无名指
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    
    # 小指
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# 手部关键点名称映射
HAND_LANDMARK_NAMES = {
    0: "wrist",
    1: "thumb_cmc", 2: "thumb_mcp", 3: "thumb_ip", 4: "thumb_tip",
    5: "index_finger_mcp", 6: "index_finger_pip", 7: "index_finger_dip", 8: "index_finger_tip",
    9: "middle_finger_mcp", 10: "middle_finger_pip", 11: "middle_finger_dip", 12: "middle_finger_tip",
    13: "ring_finger_mcp", 14: "ring_finger_pip", 15: "ring_finger_dip", 16: "ring_finger_tip",
    17: "pinky_mcp", 18: "pinky_pip", 19: "pinky_dip", 20: "pinky_tip"
}

# 手部名称到索引的反向映射
HAND_LANDMARK_INDICES = {v: k for k, v in HAND_LANDMARK_NAMES.items()}


# 手部骨骼连接定义
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # 手腕到各指根
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
    (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP),
    (HandLandmark.WRIST, HandLandmark.MIDDLE_FINGER_MCP),
    (HandLandmark.WRIST, HandLandmark.RING_FINGER_MCP),
    (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
    
    # 拇指
    (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    
    # 食指
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
    (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
    (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    
    # 中指
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
    (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
    (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    
    # 无名指
    (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
    (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
    (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    
    # 小指
    (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
    
    # 掌骨连接 (MCP之间)
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP),
]


# 合并关键点索引范围 (身体 + 双手 = 75点)
COMBINED_LANDMARK_INDICES = {
    'body': (0, 33),           # 身体: 0-32 (33个点)
    'left_hand': (33, 54),     # 左手: 33-53 (21个点)
    'right_hand': (54, 75),    # 右手: 54-74 (21个点)
}

# 总关键点数量
BODY_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
TOTAL_LANDMARK_COUNT = BODY_LANDMARK_COUNT + 2 * HAND_LANDMARK_COUNT  # 75


# 手指定义 (用于测量和手势识别)
FINGER_LANDMARKS = {
    'thumb': [HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP],
    'index': [HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP],
    'middle': [HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP],
    'ring': [HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP],
    'pinky': [HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP],
}

# 指尖索引
FINGERTIP_LANDMARKS = [
    HandLandmark.THUMB_TIP,
    HandLandmark.INDEX_FINGER_TIP,
    HandLandmark.MIDDLE_FINGER_TIP,
    HandLandmark.RING_FINGER_TIP,
    HandLandmark.PINKY_TIP,
]

# MCP关节索引 (用于掌围计算)
MCP_LANDMARKS = [
    HandLandmark.INDEX_FINGER_MCP,
    HandLandmark.MIDDLE_FINGER_MCP,
    HandLandmark.RING_FINGER_MCP,
    HandLandmark.PINKY_MCP,
]


# 手部测量所需的骨骼段
HAND_MEASUREMENT_BONES = {
    # 手掌
    'palm_length': (HandLandmark.WRIST, HandLandmark.MIDDLE_FINGER_MCP),
    'hand_width': (HandLandmark.THUMB_CMC, HandLandmark.PINKY_MCP),
    'thumb_span': (HandLandmark.THUMB_TIP, HandLandmark.INDEX_FINGER_TIP),
    
    # 拇指各段
    'thumb_cmc_mcp': (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    'thumb_mcp_ip': (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    'thumb_ip_tip': (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    
    # 食指各段
    'index_mcp_pip': (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
    'index_pip_dip': (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
    'index_dip_tip': (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    
    # 中指各段
    'middle_mcp_pip': (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
    'middle_pip_dip': (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
    'middle_dip_tip': (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    
    # 无名指各段
    'ring_mcp_pip': (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
    'ring_pip_dip': (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
    'ring_dip_tip': (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    
    # 小指各段
    'pinky_mcp_pip': (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    'pinky_pip_dip': (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    'pinky_dip_tip': (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
}


# 手部关节角度测量定义 (三个点定义一个角度)
HAND_JOINT_ANGLES = {
    # 拇指
    'thumb_mcp': (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    'thumb_ip': (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    
    # 食指
    'index_mcp': (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
    'index_pip': (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
    'index_dip': (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    
    # 中指
    'middle_mcp': (HandLandmark.WRIST, HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
    'middle_pip': (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
    'middle_dip': (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    
    # 无名指
    'ring_mcp': (HandLandmark.WRIST, HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
    'ring_pip': (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
    'ring_dip': (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    
    # 小指
    'pinky_mcp': (HandLandmark.WRIST, HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    'pinky_pip': (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    'pinky_dip': (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
}
