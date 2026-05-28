# -*- coding: utf-8 -*-
"""
UI 渲染模块

测量面板、进度条、水平仪等 UI 元素的绘制
"""
import os
import math
from typing import List, Optional, Dict, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.core.biomechanical_constraints import ConstraintsResult

# PIL 可选导入
PIL_AVAILABLE = True
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    PIL_AVAILABLE = False


def get_chinese_font(font_size: int):
    """获取中文字体"""
    if not PIL_AVAILABLE:
        return None
    font_candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for p in font_candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_progress_bar(image: np.ndarray, progress: float, x: int, y: int, 
                      width: int = 220, height: int = 16):
    """绘制进度条"""
    cv2.rectangle(image, (x, y), (x + width, y + height), (50, 50, 50), -1)
    fill_width = int(width * max(0.0, min(progress, 1.0)))
    color = (0, 255, 0) if progress >= 1.0 else (0, 255, 255)
    cv2.rectangle(image, (x, y), (x + fill_width, y + height), color, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 1)
     # Removed percentage display
    # cv2.putText(image, f"{progress*100:.0f}%", (x + width + 10, y + height - 3),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_level_indicator(image: np.ndarray, pitch: float, roll: float, 
                         x: int, y: int, radius: int = 40, threshold: float = 2.0) -> bool:
    """
    绘制水平仪指示器
    
    Returns:
        是否水平
    """
    is_level = abs(pitch) < threshold and abs(roll) < threshold
    
    # 外圈
    circle_color = (0, 255, 0) if is_level else (100, 100, 100)
    cv2.circle(image, (x, y), radius, circle_color, 2)
    cv2.circle(image, (x, y), radius // 3, (100, 100, 100), 1)
    
    # 十字线
    cv2.line(image, (x - radius, y), (x + radius, y), (100, 100, 100), 1)
    cv2.line(image, (x, y - radius), (x, y + radius), (100, 100, 100), 1)
    
    # 气泡位置
    bubble_x = int(x + roll * 2)
    bubble_y = int(y + pitch * 2)
    
    # 限制在圆内
    dist = math.sqrt((bubble_x - x)**2 + (bubble_y - y)**2)
    if dist > radius - 8:
        scale = (radius - 8) / max(dist, 0.001)
        bubble_x = int(x + (bubble_x - x) * scale)
        bubble_y = int(y + (bubble_y - y) * scale)
    
    bubble_color = (0, 255, 0) if is_level else (0, 165, 255)
    cv2.circle(image, (bubble_x, bubble_y), 10, bubble_color, -1)
    
    return is_level


def render_measurement_panel(panel_h: int, panel_w: int,
                             measurement_values_cm: Optional[Dict],
                             constraints_result: Optional['ConstraintsResult'] = None) -> np.ndarray:
    """
    渲染测量数据面板

    Args:
        panel_h: 面板高度
        panel_w: 面板宽度
        measurement_values_cm: 测量值字典
        constraints_result: 生物力学约束检查结果（可选）

    Returns:
        面板图像
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    
    if not PIL_AVAILABLE:
        cv2.putText(panel, "PIL未安装", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(panel, "请安装pillow", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return panel
    
    font_title = get_chinese_font(20)
    font_body = get_chinese_font(16)
    font_small = get_chinese_font(14)
    if font_title is None or font_body is None:
        return panel
    
    img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    padding = 10
    y = padding
    draw.text((padding, y), "测量数据", font=font_title, fill=(255, 255, 255))
    y += 30

    if not measurement_values_cm:
        draw.text((padding, y), "等待测量数据(需要深度)", font=font_body, fill=(200, 200, 200))
        y += 24
    else:
        line_h = 20
        for k, v in measurement_values_cm.items():
            if y > panel_h - 24:
                break
            if v is None:
                text = f"{k}: 无数据"
                color = (150, 150, 150)
            else:
                text = f"{k}: {v:.1f} cm"
                color = (255, 255, 255)
            draw.text((padding, y), text, font=font_body, fill=color)
            y += line_h

    draw.text((padding, panel_h - 26), "q:退出  r:重启", font=font_body, fill=(180, 180, 180))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_chinese_text(image_bgr: np.ndarray, text: str, x: int, y: int,
                       font_size: int = 16, color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    使用PIL绘制中文文本到OpenCV图像上

    Args:
        image_bgr: OpenCV BGR图像
        text: 要绘制的中文文本
        x, y: 文本位置（左上角）
        font_size: 字体大小
        color: BGR颜色元组

    Returns:
        绘制后的图像
    """
    if not PIL_AVAILABLE:
        cv2.putText(image_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image_bgr

    font = get_chinese_font(font_size)
    if font is None:
        cv2.putText(image_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image_bgr

    # 转换BGR到RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # PIL使用RGB颜色
    rgb_color = (color[2], color[1], color[0])
    draw.text((x, y), text, font=font, fill=rgb_color)

    # 转换回BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_measurement_overlay(image_bgr: np.ndarray, lines: List[str],
                             page_text: str, show_hint: str) -> np.ndarray:
    """在图像上绘制测量数据叠加层"""
    if not PIL_AVAILABLE or not lines:
        return image_bgr
    
    h, w = image_bgr.shape[:2]
    font = get_chinese_font(16)
    if font is None:
        return image_bgr
    
    padding = 8
    line_h = 20
    panel_w = 360
    panel_h = padding * 2 + line_h * (len(lines) + 2)
    
    x0 = max(10, w - panel_w - 10)
    y0 = max(10, h - panel_h - 10)
    
    overlay = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, 0, panel_w, panel_h], fill=(0, 0, 0, 140))
    
    y = padding
    draw.text((padding, y), page_text, font=font, fill=(255, 255, 255, 255))
    y += line_h
    for s in lines:
        draw.text((padding, y), s, font=font, fill=(255, 255, 255, 255))
        y += line_h
    draw.text((padding, y), show_hint, font=font, fill=(200, 200, 200, 255))
    
    base = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    base.alpha_composite(overlay, dest=(x0, y0))
    return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)
