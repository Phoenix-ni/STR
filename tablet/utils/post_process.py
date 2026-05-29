"""
推理后处理模块

对应论文 Section 3.1 后处理步骤：
    1. Split Model → 二值分割掩码 (H,) 和 (W,)
    2. 从分割掩码提取连续的分割区域
    3. 取每个分割区域的中点作为分割线
    4. 分割线组合生成 R×C 网格
    5. 网格单元 ROI 传给 Merge Model

    "Similar to SPLERGE, after obtaining the final horizontal and vertical
     split regions, the midpoint of each split region is selected as the
     position for the splitting line."
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def extract_split_lines(split_mask: np.ndarray,
                         min_gap: int = 2) -> List[float]:
    """
    从二值分割掩码提取分割线坐标

    算法：
    1. 找到所有连续的分割区域（连通子序列中值为1的部分）
    2. 取每个区域的中点作为分割线

    Args:
        split_mask: (L,) 二值数组，1=分割区域
        min_gap: 合并相邻分割区域的最小间隔（避免噪声）

    Returns:
        split_lines: 分割线坐标列表（中点坐标，float）
    """
    L = len(split_mask)
    split_lines = []

    i = 0
    while i < L:
        if split_mask[i] <= 0.5:
            i += 1
            continue

        # 找到一个“分割区域”的起止 [i, j)
        j = i
        while j < L and split_mask[j] > 0.5:
            j += 1

        # 合并紧邻的噪声分割段：若两个分割段之间的 0 间隔 <= min_gap，则视为同一段
        if min_gap is None:
            gap = 0
        else:
            gap = max(int(min_gap), 0)

        k = j
        while gap > 0 and k < L:
            # 统计接下来连续的 0 长度
            z = k
            while z < L and split_mask[z] <= 0.5:
                z += 1
            zero_len = z - k
            if zero_len == 0:
                # 已经是 1 了（理论上不会发生）
                break
            if zero_len > gap or z >= L:
                break
            # 0 间隔够小且后面还有 1 段，合并之：把 j 扩展到下一段的末尾
            k2 = z
            while k2 < L and split_mask[k2] > 0.5:
                k2 += 1
            j = k2
            k = j

        mid = (i + j - 1) / 2.0
        split_lines.append(mid)
        i = j

    return split_lines


def split_lines_to_intervals(split_lines: List[float],
                               total_length: int,
                               table_start: float = None,
                               table_end: float = None) -> List[Tuple[float, float]]:
    """
    将分割线坐标转换为区间列表（每行/列的范围）

    Args:
        split_lines: 分割线坐标（有序），应包含首尾边界
        total_length: 总长度（图像 H 或 W）
        table_start: 表格起始坐标（可选，用于裁剪）
        table_end: 表格结束坐标（可选，用于裁剪）

    Returns:
        intervals: [(start, end), ...] 每行/列的范围
    """
    if len(split_lines) < 2:
        # 无法划分，返回整个范围
        start = table_start if table_start is not None else 0
        end = table_end if table_end is not None else total_length
        return [(start, end)]

    intervals = []
    for i in range(len(split_lines) - 1):
        start = split_lines[i]
        end = split_lines[i + 1]
        if end > start:
            intervals.append((start, end))

    return intervals


def split_masks_to_grid(row_splits: np.ndarray,
                         col_splits: np.ndarray,
                         img_h: int = 960,
                         img_w: int = 960,
                         table_bbox: Optional[Tuple[float, float, float, float]] = None
                         ) -> Tuple[List[Tuple[float, float]],
                                    List[Tuple[float, float]],
                                    np.ndarray]:
    """
    从行/列分割掩码生成网格单元 ROI

    对应论文 Section 3.1 推理后处理：
        "the midpoint of each split region is selected as the position
         for the splitting line. The combination of horizontal and vertical
         split lines divides the table image into a grid structure with R rows and C columns."

    Args:
        row_splits: (H,) 行分割掩码（H分辨率）
        col_splits: (W,) 列分割掩码（W分辨率）
        img_h: 图像高度（960）
        img_w: 图像宽度（960）
        table_bbox: 表格边界框 (x1, y1, x2, y2)，用于确定表格区域

    Returns:
        rows: [(ymin, ymax), ...] 行范围列表（R行）
        cols: [(xmin, xmax), ...] 列范围列表（C列）
        cell_boxes: (R*C, 4) ROI 坐标，格式 (x1, y1, x2, y2)
    """
    # 提取分割线
    row_lines = extract_split_lines(row_splits)
    col_lines = extract_split_lines(col_splits)

    # 确定表格边界
    if table_bbox is not None:
        t_x1, t_y1, t_x2, t_y2 = table_bbox
    else:
        t_x1, t_y1 = 0.0, 0.0
        t_x2, t_y2 = float(img_w), float(img_h)

    # 确保首尾分割线包含表格边界
    if not row_lines or row_lines[0] > t_y1 + 5:
        row_lines = [t_y1] + row_lines
    if not row_lines or row_lines[-1] < t_y2 - 5:
        row_lines = row_lines + [t_y2]

    if not col_lines or col_lines[0] > t_x1 + 5:
        col_lines = [t_x1] + col_lines
    if not col_lines or col_lines[-1] < t_x2 - 5:
        col_lines = col_lines + [t_x2]

    # 生成行/列区间
    rows = [(row_lines[i], row_lines[i+1])
            for i in range(len(row_lines)-1)
            if row_lines[i+1] > row_lines[i]]

    cols = [(col_lines[i], col_lines[i+1])
            for i in range(len(col_lines)-1)
            if col_lines[i+1] > col_lines[i]]

    R = len(rows)
    C = len(cols)

    if R == 0 or C == 0:
        return rows, cols, np.zeros((0, 4), dtype=np.float32)

    # 生成网格单元 ROI（行优先顺序）
    cell_boxes = []
    for r in range(R):
        ymin, ymax = rows[r]
        for c in range(C):
            xmin, xmax = cols[c]
            cell_boxes.append([xmin, ymin, xmax, ymax])

    cell_boxes = np.array(cell_boxes, dtype=np.float32)

    return rows, cols, cell_boxes


def apply_ocr_post_processing(row_splits: np.ndarray,
                                col_splits: np.ndarray,
                                word_bboxes: List[dict],
                                scale: float,
                                pad_top: int,
                                pad_left: int,
                                img_h: int = 960) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用 OCR 文本投影后处理（论文 Section 3.1 的可选步骤）

    对应论文：
        "All text blocks within the table are first extracted using OCR.
         The center point of each text block is then projected along the
         horizontal/vertical directions. If any non-split regions in the
         horizontal/vertical direction contain no text projection, they are
         reclassified as split regions."

    Args:
        row_splits: (H,) 行分割掩码
        col_splits: (W,) 列分割掩码
        word_bboxes: OCR 词框列表，每个含 'bbox' (x1,y1,x2,y2) 和 'text'
        scale: 图像缩放比例
        pad_top: 顶部填充
        pad_left: 左侧填充
        img_h: 图像高度（960）

    Returns:
        row_splits_refined: 修正后的行分割掩码
        col_splits_refined: 修正后的列分割掩码
    """
    H = len(row_splits)
    W = len(col_splits)

    row_splits_refined = row_splits.copy()
    col_splits_refined = col_splits.copy()

    if not word_bboxes:
        return row_splits_refined, col_splits_refined

    # 计算文本投影点（中心点）
    row_projections = set()
    col_projections = set()

    for word in word_bboxes:
        bbox = word['bbox']
        # 原始坐标变换到处理后的图像空间
        cx = ((bbox[0] + bbox[2]) / 2) * scale + pad_left
        cy = ((bbox[1] + bbox[3]) / 2) * scale + pad_top

        if 0 <= int(cy) < H:
            row_projections.add(int(cy))
        if 0 <= int(cx) < W:
            col_projections.add(int(cx))

    # 对每个非分割区域，若不含文本投影，则将其标记为分割区域
    # 这处理了空行/列的情况
    # 找到现有分割区域的边界
    def find_regions(mask):
        """找到所有连续的非分割区域"""
        regions = []
        in_non_split = False
        start = 0
        for i in range(len(mask)):
            if mask[i] < 0.5 and not in_non_split:
                in_non_split = True
                start = i
            elif mask[i] > 0.5 and in_non_split:
                in_non_split = False
                regions.append((start, i))
        if in_non_split:
            regions.append((start, len(mask)))
        return regions

    # 行方向后处理
    non_split_regions = find_regions(row_splits_refined)
    for start, end in non_split_regions:
        # 跳过表格头尾区域
        if start == 0 or end == H:
            continue
        # 检查区域中是否有文本投影
        has_text = any(start <= p < end for p in row_projections)
        if not has_text:
            # 无文本，标记为分割区域
            mid = (start + end) // 2
            row_splits_refined[max(0, mid-2):min(H, mid+3)] = 1.0

    # 列方向后处理
    non_split_regions = find_regions(col_splits_refined)
    for start, end in non_split_regions:
        if start == 0 or end == W:
            continue
        has_text = any(start <= p < end for p in col_projections)
        if not has_text:
            mid = (start + end) // 2
            col_splits_refined[max(0, mid-2):min(W, mid+3)] = 1.0

    return row_splits_refined, col_splits_refined
