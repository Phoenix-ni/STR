"""
图像预处理模块

对应论文 Section 4.1：
    "the longer side of each image is resized to 960 pixels while maintaining
     the original aspect ratio. Then, blank padding is added as needed to ensure
     that both the height (H) and width (W) are 960 pixels."

    对于分割注释区域：
    "if the original width is less than 5 pixels, the region is expanded based
     on its midpoint to ensure a minimum width of 5 pixels."
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def resize_and_pad(image: np.ndarray, target_size: int = 960) -> Tuple[np.ndarray, float, int, int]:
    """
    将图像等比缩放至最长边为 target_size，然后补0使两边均为 target_size

    Args:
        image: 输入图像 (H, W, 3)
        target_size: 目标尺寸（论文中为960）

    Returns:
        padded_image: (target_size, target_size, 3) 预处理后的图像
        scale: 缩放比例（用于坐标变换）
        pad_top: 顶部填充像素数
        pad_left: 左侧填充像素数
    """
    h, w = image.shape[:2]

    # 按最长边缩放
    if h > w:
        scale = target_size / h
    else:
        scale = target_size / w

    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    # 双线性插值缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充量（居中填充）
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    # 填充0
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    return padded, scale, pad_top, pad_left


def transform_bbox(bbox: Tuple[float, float, float, float],
                   scale: float, pad_top: int, pad_left: int) -> Tuple[float, float, float, float]:
    """
    将边界框坐标从原始空间变换到预处理后的图像空间

    Args:
        bbox: (xmin, ymin, xmax, ymax) 原始坐标
        scale: 缩放比例
        pad_top: 顶部填充
        pad_left: 左侧填充

    Returns:
        (xmin, ymin, xmax, ymax) 变换后的坐标
    """
    xmin, ymin, xmax, ymax = bbox
    xmin = xmin * scale + pad_left
    ymin = ymin * scale + pad_top
    xmax = xmax * scale + pad_left
    ymax = ymax * scale + pad_top
    return xmin, ymin, xmax, ymax


def transform_bboxes(bboxes: List[Tuple[float, float, float, float]],
                     scale: float, pad_top: int,
                     pad_left: int) -> List[Tuple[float, float, float, float]]:
    """批量变换边界框坐标"""
    return [transform_bbox(b, scale, pad_top, pad_left) for b in bboxes]


def generate_split_mask_at_half_res(row_lines: List[float],
                                     col_lines: List[float],
                                     img_h: int = 960,
                                     img_w: int = 960,
                                     min_width: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    从行/列分割线生成 H/2 分辨率的二值 Split Mask

    对应论文 Section 3.1 + 4.1：
        - Split model 在 H/2 分辨率预测，2×上采样到 H
        - 最小宽度 5 像素（960空间）
        - 分割区域应对齐“真实分割线(行/列边界)”附近的一小段区域

    重要：row_lines/col_lines 为“边界线”（首尾 = 表格外框，中间 = 行/列分隔线）。
    训练时监督所有边界（含外边框），与 XML bndbox 提取的框线一致。

    注意：row_lines 是在960分辨率下的y坐标（表格行边界）
         col_lines 是在960分辨率下的x坐标（表格列边界）

    Args:
        row_lines: 水平分割线的 y 坐标列表 (在960空间)，已排序
        col_lines: 垂直分割线的 x 坐标列表 (在960空间)，已排序
        img_h: 图像高度（960）
        img_w: 图像宽度（960）
        min_width: 分割区域最小宽度（像素，960空间）

    Returns:
        row_mask_half: (H/2,) 行分割 mask，1=分割区域（H/2=480分辨率）
        col_mask_half: (W/2,) 列分割 mask，1=分割区域（W/2=480分辨率）
    """
    half_h = img_h // 2    # 480
    half_w = img_w // 2    # 480

    row_mask_full = np.zeros(img_h, dtype=np.float32)
    col_mask_full = np.zeros(img_w, dtype=np.float32)

    # 将“边界线”转换为“线附近的分割区域”（含表格外边框）
    # 监督所有边界：首尾 = 表格外框，中间 = 行/列分隔线。
    half_w = max(int(min_width) // 2, 1)

    if len(row_lines) >= 1:
        for y in row_lines:
            y_i = int(round(float(y)))
            lo = max(0, y_i - half_w)
            hi = min(img_h, y_i + half_w + 1)
            if hi > lo:
                row_mask_full[lo:hi] = 1.0

    if len(col_lines) >= 1:
        for x in col_lines:
            x_i = int(round(float(x)))
            lo = max(0, x_i - half_w)
            hi = min(img_w, x_i + half_w + 1)
            if hi > lo:
                col_mask_full[lo:hi] = 1.0

    # 降采样到 H/2：取每2个像素的最大值（若有任一为1则为1）
    row_mask_half = np.maximum(
        row_mask_full[0::2], row_mask_full[1::2]
    )  # (H/2,)

    col_mask_half = np.maximum(
        col_mask_full[0::2], col_mask_full[1::2]
    )  # (W/2,)

    return row_mask_half, col_mask_half


def extract_grid_lines_from_annotations(rows: List[Tuple[float, float]],
                                         cols: List[Tuple[float, float]],
                                         table_bbox: Tuple[float, float, float, float]
                                         ) -> Tuple[List[float], List[float]]:
    """
    从行/列标注中提取分割线坐标（用于 Split Mask 生成）

    分割线定义：
    - 第一行顶部 + 每行底部 = 所有行间水平分割线
    - 第一列左侧 + 每列右侧 = 所有列间垂直分割线

    Args:
        rows: 行标注 [(ymin, ymax), ...] 已按 ymin 排序
        cols: 列标注 [(xmin, xmax), ...] 已按 xmin 排序
        table_bbox: (xmin, ymin, xmax, ymax) 表格边界框

    Returns:
        row_lines: 水平分割线 y 坐标列表
        col_lines: 垂直分割线 x 坐标列表
    """
    row_lines = []
    if rows:
        # 第一条线：表格顶边 / 第一行顶边
        row_lines.append(rows[0][0])
        # 每行底边
        for _, ymax in rows:
            row_lines.append(ymax)

    col_lines = []
    if cols:
        # 第一条线：表格左边 / 第一列左边
        col_lines.append(cols[0][0])
        # 每列右边
        for _, xmax in cols:
            col_lines.append(xmax)

    return row_lines, col_lines


def compute_grid_cell_boxes(rows: List[Tuple[float, float]],
                             cols: List[Tuple[float, float]]) -> np.ndarray:
    """
    计算网格单元的边界框（按行优先顺序）

    Args:
        rows: 行范围 [(ymin, ymax), ...]
        cols: 列范围 [(xmin, xmax), ...]

    Returns:
        cell_boxes: (R*C, 4) ndarray，格式 (x1, y1, x2, y2)
    """
    R = len(rows)
    C = len(cols)
    boxes = []
    for r in range(R):
        for c in range(C):
            ymin, ymax = rows[r]
            xmin, xmax = cols[c]
            boxes.append([xmin, ymin, xmax, ymax])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    图像归一化到 [0, 1]

    Returns:
        归一化后的图像 float32
    """
    return image.astype(np.float32) / 255.0


# ImageNet 均值和标准差（用于迁移学习）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    """
    使用 ImageNet 均值/方差归一化

    Args:
        image: (H, W, 3) float32，值范围 [0, 1]

    Returns:
        normalized: (H, W, 3) float32
    """
    return (image - IMAGENET_MEAN) / IMAGENET_STD
