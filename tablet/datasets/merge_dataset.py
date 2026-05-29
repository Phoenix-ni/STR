"""
Merge Model 数据集

每个样本包含：
- 预处理后的表格图像 (3, H, W)
- 网格单元 ROI 边界框 (R*C, 4)，x1y1x2y2格式
- OTSL 标签序列 (R*C,)，整数编码
- 网格形状 (R, C)
"""

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Dict

from datasets.xml_parser import parse_fintabnet_xml
from datasets.preprocess import (
    resize_and_pad,
    compute_grid_cell_boxes,
    normalize_image,
    normalize_imagenet
)


# OTSL 字符串到整数 ID 的映射
OTSL_LABEL_MAP = {'C': 0, 'L': 1, 'U': 2, 'X': 3}
OTSL_ID2LABEL = {0: 'C', 1: 'L', 2: 'U', 3: 'X'}


class MergeDataset(Dataset):
    """
    Merge Model 训练/验证数据集

    结合 XML 标注（提取网格单元位置）和预计算 OTSL 标签使用

    Args:
        xml_dir: XML 标注目录
        otsl_dir: 预计算 OTSL 标签目录（JSON文件）
        images_dir: 图像目录
        target_size: 目标图像尺寸（960）
        max_seq_len: 最大序列长度（640），超过则截断
        augment: 是否进行数据增强
    """

    def __init__(self,
                 xml_dir: str,
                 otsl_dir: str,
                 images_dir: str,
                 target_size: int = 960,
                 max_seq_len: int = 640,
                 augment: bool = False):
        super().__init__()

        self.xml_dir = xml_dir
        self.otsl_dir = otsl_dir
        self.images_dir = images_dir
        self.target_size = target_size
        self.max_seq_len = max_seq_len
        self.augment = augment

        # 收集所有有效样本
        self.samples = self._collect_samples()
        print(f"[MergeDataset] 加载 {len(self.samples)} 个样本 from {xml_dir}")

    def _collect_samples(self) -> List[Dict]:
        """收集所有有效的三元组 (xml_path, otsl_path, image_path)"""
        samples = []

        for xml_file in sorted(os.listdir(self.xml_dir)):
            if not xml_file.endswith('.xml'):
                continue

            base_name = os.path.splitext(xml_file)[0]
            xml_path = os.path.join(self.xml_dir, xml_file)
            img_path = os.path.join(self.images_dir, base_name + '.jpg')
            otsl_path = os.path.join(self.otsl_dir, base_name + '_otsl.json')

            # 三个文件均需存在
            if not os.path.exists(img_path):
                continue
            if not os.path.exists(otsl_path):
                continue

            samples.append({
                'xml_path': xml_path,
                'img_path': img_path,
                'otsl_path': otsl_path,
                'base_name': base_name
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            {
                'image': (3, H, W) float32 tensor
                'rois': (N, 4) float32 tensor，网格单元 ROI (x1,y1,x2,y2) 绝对坐标
                'otsl_labels': (N,) int64 tensor，OTSL 类别 id
                'grid_shape': (R, C) tuple
                'num_cells': int = R * C
                'filename': str
            }
        """
        sample = self.samples[idx]

        # 1. 解析 XML
        ann = parse_fintabnet_xml(sample['xml_path'])

        # 2. 加载 OTSL 标签
        with open(sample['otsl_path'], 'r') as f:
            otsl_data = json.load(f)

        otsl_sequence = otsl_data['otsl_sequence']   # List[str]
        grid_shape = tuple(otsl_data['grid_shape'])  # (R, C)
        R, C = grid_shape

        # 3. 读取图像
        image = cv2.imread(sample['img_path'])
        if image is None:
            image = np.zeros((100, 100, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. 预处理：resize + pad 到 target_size × target_size
        image, scale, pad_top, pad_left = resize_and_pad(image, self.target_size)

        # 5. 变换行/列坐标到新空间
        rows_scaled = [
            (y0 * scale + pad_top, y1 * scale + pad_top)
            for y0, y1 in ann['rows']
        ]
        cols_scaled = [
            (x0 * scale + pad_left, x1 * scale + pad_left)
            for x0, x1 in ann['cols']
        ]

        # 6. 计算网格单元 ROI 边界框（行优先顺序）
        # 使用 XML 中的行/列标注（可能比 OTSL 来源更精确）
        # 若行/列数量与 OTSL 网格不一致，使用 OTSL 的网格形状
        if len(rows_scaled) == R and len(cols_scaled) == C:
            cell_boxes = compute_grid_cell_boxes(rows_scaled, cols_scaled)
        else:
            # 回退：从 OTSL 数据中尝试获取位置信息，或使用均匀划分
            cell_boxes = self._fallback_grid_boxes(
                ann, rows_scaled, cols_scaled, R, C, scale, pad_top, pad_left
            )

        # 7. 编码 OTSL 标签为整数
        n_cells = R * C
        otsl_ids = np.array(
            [OTSL_LABEL_MAP.get(t, 0) for t in otsl_sequence[:n_cells]],
            dtype=np.int64
        )

        # 截断到 max_seq_len
        if n_cells > self.max_seq_len:
            cell_boxes = cell_boxes[:self.max_seq_len]
            otsl_ids = otsl_ids[:self.max_seq_len]
            n_cells = self.max_seq_len

        # 8. 数据增强
        if self.augment:
            image = self._augment_image(image)

        # 9. 归一化图像
        image_f = normalize_image(image)
        image_f = normalize_imagenet(image_f)
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1))   # (3, H, W)

        rois_t = torch.from_numpy(cell_boxes)        # (N, 4)
        otsl_t = torch.from_numpy(otsl_ids)          # (N,)

        # ============================================================
        # 数据有效性检查（防止训练时出现NaN）
        # ============================================================
        
        # 检查ROI有效性
        assert rois_t.shape[0] > 0, f"[{sample['base_name']}] No ROIs"
        assert (rois_t >= 0).all(), f"[{sample['base_name']}] Negative ROI coordinates"
        assert (rois_t[:, 0] <= rois_t[:, 2]).all(), f"[{sample['base_name']}] Invalid x coordinates (x1>x2)"
        assert (rois_t[:, 1] <= rois_t[:, 3]).all(), f"[{sample['base_name']}] Invalid y coordinates (y1>y2)"
        
        # 检查坐标范围（应该在[0, target_size]范围内）
        max_coord = self.target_size
        assert (rois_t <= max_coord).all(), f"[{sample['base_name']}] ROI coordinates exceed image size"
        
        # 检查ROI不为零大小
        roi_areas = (rois_t[:, 2] - rois_t[:, 0]) * (rois_t[:, 3] - rois_t[:, 1])
        assert (roi_areas > 0).all(), f"[{sample['base_name']}] Found zero-area ROIs"
        
        # 检查标签
        assert otsl_t.shape[0] == rois_t.shape[0], f"[{sample['base_name']}] Label/ROI count mismatch"
        assert (otsl_t >= 0).all() and (otsl_t < 4).all(), f"[{sample['base_name']}] Invalid OTSL label"
        
        # 检查图像有效性
        assert image_t.shape == (3, self.target_size, self.target_size), \
            f"[{sample['base_name']}] Image shape mismatch"
        assert not torch.isnan(image_t).any(), f"[{sample['base_name']}] NaN in image"
        assert not torch.isinf(image_t).any(), f"[{sample['base_name']}] Inf in image"

        return {
            'image': image_t,
            'rois': rois_t,
            'otsl_labels': otsl_t,
            'grid_shape': grid_shape,
            'num_cells': n_cells,
            'filename': sample['base_name']
        }

    def _fallback_grid_boxes(self, ann, rows_scaled, cols_scaled,
                              R, C, scale, pad_top, pad_left) -> np.ndarray:
        """
        行/列数量不匹配时的回退策略：
        尝试使用表格边界框均匀划分
        """
        if ann['table_bbox'] is not None:
            xmin, ymin, xmax, ymax = ann['table_bbox']
            xmin = xmin * scale + pad_left
            ymin = ymin * scale + pad_top
            xmax = xmax * scale + pad_left
            ymax = ymax * scale + pad_top

            row_height = (ymax - ymin) / R
            col_width = (xmax - xmin) / C

            rows_uniform = [(ymin + r * row_height, ymin + (r + 1) * row_height) for r in range(R)]
            cols_uniform = [(xmin + c * col_width, xmin + (c + 1) * col_width) for c in range(C)]
            return compute_grid_cell_boxes(rows_uniform, cols_uniform)
        else:
            # 最终回退：使用整张图像均匀划分
            H = W = self.target_size
            row_height = H / R
            col_width = W / C
            rows_uniform = [(r * row_height, (r + 1) * row_height) for r in range(R)]
            cols_uniform = [(c * col_width, (c + 1) * col_width) for c in range(C)]
            return compute_grid_cell_boxes(rows_uniform, cols_uniform)

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """简单图像增强"""
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return image


def merge_collate_fn(batch: List[dict]) -> dict:
    """
    Merge Dataset 的 collate 函数

    由于每个样本的 R*C 不同，需要 padding 处理
    """
    # 图像 stack
    images = torch.stack([item['image'] for item in batch])

    # ROI 和标签单独收集（不同长度）
    rois_list = [item['rois'] for item in batch]
    otsl_labels_list = [item['otsl_labels'] for item in batch]
    grid_shapes = [item['grid_shape'] for item in batch]
    num_cells = [item['num_cells'] for item in batch]
    filenames = [item['filename'] for item in batch]

    # 对 OTSL 标签进行 padding（用 -100 表示忽略）
    max_cells = max(num_cells)
    otsl_padded = torch.full((len(batch), max_cells), fill_value=-100, dtype=torch.long)
    for i, (labels, n) in enumerate(zip(otsl_labels_list, num_cells)):
        otsl_padded[i, :n] = labels

    return {
        'image': images,
        'rois_list': rois_list,       # List[Tensor(N_i, 4)]
        'otsl_labels': otsl_padded,   # (B, max_n) padded with -100
        'grid_shapes': grid_shapes,   # List[(R_i, C_i)]
        'num_cells': num_cells,
        'filename': filenames
    }
