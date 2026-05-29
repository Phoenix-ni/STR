"""
Split Model 数据集

每个样本包含：
- 预处理后的表格图像 (3, H, W)
- 行分割标签 (H/2,)，0/1 二值
- 列分割标签 (W/2,)，0/1 二值
"""

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List

from datasets.xml_parser import parse_fintabnet_xml, get_row_col_split_lines
from datasets.preprocess import (
    resize_and_pad,
    transform_bboxes,
    generate_split_mask_at_half_res,
    normalize_image,
    normalize_imagenet,
    IMAGENET_MEAN,
    IMAGENET_STD
)


class SplitDataset(Dataset):
    """
    Split Model 训练/验证数据集

    从 XML 标注动态生成分割标签（不依赖预计算的 npy 文件）

    Args:
        xml_dir: XML 标注目录（train/val/test）
        images_dir: 图像目录
        target_size: 目标图像尺寸（960）
        min_split_width: 分割区域最小宽度（5像素）
        augment: 是否进行数据增强
    """

    def __init__(self,
                 xml_dir: str,
                 images_dir: str,
                 target_size: int = 960,
                 min_split_width: int = 5,
                 augment: bool = False):
        super().__init__()

        self.xml_dir = xml_dir
        self.images_dir = images_dir
        self.target_size = target_size
        self.min_split_width = min_split_width
        self.augment = augment

        # 收集所有有效样本
        self.samples = self._collect_samples()
        print(f"[SplitDataset] 加载 {len(self.samples)} 个样本 from {xml_dir}")

    def _collect_samples(self) -> List[Tuple[str, str]]:
        """收集所有有效的 (xml_path, image_path) 对"""
        samples = []
        for xml_file in sorted(os.listdir(self.xml_dir)):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(self.xml_dir, xml_file)

            # 图像文件名与XML相同（.jpg格式）
            base_name = os.path.splitext(xml_file)[0]
            img_path = os.path.join(self.images_dir, base_name + '.jpg')

            if not os.path.exists(img_path):
                continue

            samples.append((xml_path, img_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            {
                'image': (3, H, W) float32 tensor
                'row_mask': (H/2,) float32 tensor，0/1 分割标签
                'col_mask': (W/2,) float32 tensor，0/1 分割标签
                'num_rows': int 表格行数
                'num_cols': int 表格列数
                'filename': str
            }
        """
        xml_path, img_path = self.samples[idx]

        # 解析 XML
        ann = parse_fintabnet_xml(xml_path)

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            # fallback：返回空图像
            image = np.zeros((100, 100, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 预处理：resize + pad 到 target_size × target_size
        image, scale, pad_top, pad_left = resize_and_pad(image, self.target_size)

        # 变换行/列坐标到新空间
        rows_scaled = [
            (y0 * scale + pad_top, y1 * scale + pad_top)
            for y0, y1 in ann['rows']
        ]
        cols_scaled = [
            (x0 * scale + pad_left, x1 * scale + pad_left)
            for x0, x1 in ann['cols']
        ]

        # 生成分割线坐标
        row_lines, col_lines = get_row_col_split_lines(rows_scaled, cols_scaled)

        # 生成分割掩码（H/2 分辨率）
        row_mask, col_mask = generate_split_mask_at_half_res(
            row_lines, col_lines,
            img_h=self.target_size,
            img_w=self.target_size,
            min_width=self.min_split_width
        )

        # 数据增强（仅训练时）
        if self.augment:
            image, row_mask, col_mask = self._augment(image, row_mask, col_mask)

        # 归一化图像：[0,255] → [0,1] → ImageNet 标准化
        image_f = normalize_image(image)          # (H, W, 3) float32
        image_f = normalize_imagenet(image_f)     # ImageNet 标准化
        image_t = torch.from_numpy(
            image_f.transpose(2, 0, 1)            # (3, H, W)
        )

        row_mask_t = torch.from_numpy(row_mask)   # (H/2,)
        col_mask_t = torch.from_numpy(col_mask)   # (W/2,)

        return {
            'image': image_t,
            'row_mask': row_mask_t,
            'col_mask': col_mask_t,
            'num_rows': len(ann['rows']),
            'num_cols': len(ann['cols']),
            'filename': os.path.splitext(os.path.basename(xml_path))[0]
        }

    def _augment(self, image: np.ndarray,
                 row_mask: np.ndarray,
                 col_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        简单数据增强：随机亮度/对比度调整
        （论文未明确说明使用增强，但在工业实践中通常会用到）
        """
        # 随机亮度调整
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # 随机对比度调整
        if np.random.rand() < 0.3:
            mean = image.mean()
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(mean + (image.astype(np.float32) - mean) * factor, 0, 255).astype(np.uint8)

        return image, row_mask, col_mask


def split_collate_fn(batch: List[dict]) -> dict:
    """
    Split Dataset 的 collate 函数

    将样本列表组合为 batch 字典
    """
    images = torch.stack([item['image'] for item in batch])
    row_masks = torch.stack([item['row_mask'] for item in batch])
    col_masks = torch.stack([item['col_mask'] for item in batch])
    num_rows = [item['num_rows'] for item in batch]
    num_cols = [item['num_cols'] for item in batch]
    filenames = [item['filename'] for item in batch]

    return {
        'image': images,
        'row_mask': row_masks,
        'col_mask': col_masks,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'filename': filenames
    }
