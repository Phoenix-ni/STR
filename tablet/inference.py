"""
TABLET 完整推理流水线

对应论文 System Framework (Figure 2)：
    1. 图像预处理 (resize + pad to 960×960)
    2. Split Model → 行/列二值分割掩码
    3. 后处理：分割掩码 → 网格 (R×C)，提取每个单元格 ROI
    4. Merge Model → OTSL 分类 (C/L/U/X)
    5. OTSL → HTML
    6. OCR 文本填入单元格

使用方法：
    python inference.py --image table.jpg \
        --split-checkpoint checkpoints/split/best.pth \
        --merge-checkpoint checkpoints/merge/best.pth

    python inference.py --data-root ./data/fintabnet/FinTabNet.c-Structure \
        --split checkpoints/split/best.pth \
        --merge checkpoints/merge/best.pth \
        --output-dir ./outputs/predictions \
        --split test
"""

import os
import sys
import argparse
import json
import time
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from models.split_model import SplitModel
from models.merge_model import MergeModel
from datasets.preprocess import (
    resize_and_pad, normalize_image, normalize_imagenet
)
from utils.post_process import split_masks_to_grid
from utils.otsl_utils import otsl_to_html, otsl_sequence_to_spans
from configs.config import SplitConfig, MergeConfig, DataConfig

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TABLETInference:
    """
    TABLET 完整推理流水线

    支持单张或批量表格图像处理
    """

    def __init__(self,
                 split_checkpoint: str,
                 merge_checkpoint: str,
                 device: str = 'cuda',
                 img_size: int = 960,
                 split_threshold: float = 0.5):
        """
        Args:
            split_checkpoint: Split Model 权重路径
            merge_checkpoint: Merge Model 权重路径
            device: 推理设备
            img_size: 图像目标尺寸 (论文: 960)
            split_threshold: 分割二值化阈值
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.split_threshold = split_threshold

        logger.info(f"初始化 TABLET 推理流水线，设备: {self.device}")

        # 加载 Split Model
        self.split_model = self._load_split_model(split_checkpoint)

        # 加载 Merge Model
        self.merge_model = self._load_merge_model(merge_checkpoint)

    def _load_split_model(self, checkpoint_path: str) -> SplitModel:
        """加载 Split Model"""
        model = SplitModel(
            img_h=self.img_size,
            img_w=self.img_size,
            fpn_channels=SplitConfig.fpn_out_channels,
            transformer_layers=SplitConfig.transformer_layers,
            transformer_heads=SplitConfig.transformer_heads,
            transformer_ffn_dim=SplitConfig.transformer_ffn_dim,
            transformer_dropout=SplitConfig.transformer_dropout
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Split Model 加载完成: {checkpoint_path}")
        else:
            logger.warning(f"Split Model 检查点不存在: {checkpoint_path}，使用随机初始化")

        model = model.to(self.device)
        model.eval()
        return model

    def _load_merge_model(self, checkpoint_path: str) -> MergeModel:
        """加载 Merge Model"""
        model = MergeModel(
            img_h=self.img_size,
            img_w=self.img_size,
            fpn_channels=MergeConfig.fpn_out_channels,
            roi_output_size=MergeConfig.roi_output_size,
            mlp_hidden_dim=MergeConfig.mlp_hidden_dim,
            mlp_output_dim=MergeConfig.mlp_output_dim,
            transformer_layers=MergeConfig.transformer_layers,
            transformer_heads=MergeConfig.transformer_heads,
            transformer_ffn_dim=MergeConfig.transformer_ffn_dim,
            transformer_dropout=MergeConfig.transformer_dropout,
            max_rows=MergeConfig.max_rows,
            max_cols=MergeConfig.max_cols,
            max_seq_len=MergeConfig.max_seq_len,
            num_classes=MergeConfig.num_classes
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Merge Model 加载完成: {checkpoint_path}")
        else:
            logger.warning(f"Merge Model 检查点不存在: {checkpoint_path}，使用随机初始化")

        model = model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, float, int, int]:
        """
        图像预处理

        对应论文 Section 4.1：resize + pad to 960×960
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize + pad
        processed, scale, pad_top, pad_left = resize_and_pad(image_rgb, self.img_size)

        # 归一化
        image_f = normalize_image(processed)
        image_f = normalize_imagenet(image_f)
        image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)

        return image_t, processed, scale, pad_top, pad_left

    @torch.no_grad()
    def run_split(self, image_t: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行 Split Model

        Returns:
            row_splits: (H,) 二值行分割掩码
            col_splits: (W,) 二值列分割掩码
        """
        image_t = image_t.to(self.device)
        row_logits, col_logits = self.split_model(image_t)  # (1, H), (1, W)

        row_probs = torch.sigmoid(row_logits[0]).cpu().numpy()   # (H,)
        col_probs = torch.sigmoid(col_logits[0]).cpu().numpy()   # (W,)

        row_splits = (row_probs > self.split_threshold).astype(np.float32)
        col_splits = (col_probs > self.split_threshold).astype(np.float32)

        return row_splits, col_splits

    @torch.no_grad()
    def run_merge(self, image_t: torch.Tensor,
                   cell_boxes: np.ndarray,
                   grid_shape: Tuple[int, int]) -> List[str]:
        """
        运行 Merge Model

        Returns:
            otsl_sequence: OTSL token 列表
        """
        if len(cell_boxes) == 0 or grid_shape[0] == 0 or grid_shape[1] == 0:
            return []

        image_t = image_t.to(self.device)
        cells_t = torch.from_numpy(cell_boxes).to(self.device)

        logits, padding_mask = self.merge_model(
            image_t,
            [cells_t],
            [grid_shape]
        )

        # 取第一个（且唯一的）样本的预测
        R, C = grid_shape
        n_cells = min(R * C, MergeConfig.max_seq_len)
        pred_ids = logits[0, :n_cells].argmax(dim=-1).cpu().tolist()

        id2label = MergeConfig.OTSL_ID2LABEL
        otsl_sequence = [id2label[i] for i in pred_ids]

        return otsl_sequence

    def process_single_image(self,
                              image_path: str,
                              word_bboxes: Optional[List[dict]] = None
                              ) -> Dict:
        """
        处理单张表格图像

        对应论文 System Framework Figure 2 的完整流程

        Args:
            image_path: 表格图像路径
            word_bboxes: OCR 词框（可选，用于文本后处理和填充）

        Returns:
            {
                'html': str,               最终 HTML
                'otsl_sequence': List[str], OTSL 序列
                'grid_shape': (R, C),      网格形状
                'rows': [...],             行范围
                'cols': [...],             列范围
                'cell_boxes': ndarray,     单元格 ROI
                'processing_time': float,  处理耗时
            }
        """
        t0 = time.time()

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # Step 1: 图像预处理
        image_t, _processed_rgb, scale, pad_top, pad_left = self.preprocess_image(image)

        # Step 2: Split Model → 行/列分割掩码
        row_splits, col_splits = self.run_split(image_t)

        # Step 3: 后处理：分割掩码 → 网格单元 ROI
        rows, cols, _ = split_masks_to_grid(
            row_splits, col_splits,
            img_h=self.img_size,
            img_w=self.img_size
        )

        # 去掉最外侧一圈“表格外边框线”对应的行/列（一致性处理，同 test.py）
        if len(rows) > 2 and len(cols) > 2:
            rows = rows[1:-1]
            cols = cols[1:-1]

        R, C = len(rows), len(cols)
        grid_shape = (R, C)

        # 重新构造内部网格的 cell_boxes（基于去掉外边框后的内部网格）
        cell_boxes_list = []
        for (y1, y2) in rows:
            for (x1, x2) in cols:
                cell_boxes_list.append([x1, y1, x2, y2])
        cell_boxes = np.array(cell_boxes_list, dtype=np.float32)

        if R == 0 or C == 0:
            logger.warning(f"未能提取有效网格: {image_path}")
            return {
                'html': '<html><body><table></table></body></html>',
                'otsl_sequence': [],
                'grid_shape': (0, 0),
                'rows': [], 'cols': [],
                'cell_boxes': np.array([]),
                'scale': scale, 'pad_top': pad_top, 'pad_left': pad_left,
                'processing_time': time.time() - t0
            }

        # Step 4: Merge Model → OTSL 分类
        otsl_sequence = self.run_merge(image_t, cell_boxes, grid_shape)

        # Step 5: OTSL → HTML
        # 获取单元格文本内容（来自 OCR）
        cell_contents = None
        if word_bboxes:
            cell_contents = self._assign_words_to_cells(
                word_bboxes, rows, cols, scale, pad_top, pad_left, otsl_sequence, grid_shape
            )

        html = otsl_to_html(otsl_sequence, grid_shape, cell_contents)

        processing_time = time.time() - t0

        return {
            'html': html,
            'otsl_sequence': otsl_sequence,
            'grid_shape': grid_shape,
            'rows': rows,
            'cols': cols,
            'cell_boxes': cell_boxes,
            'scale': scale, 'pad_top': pad_top, 'pad_left': pad_left,
            'processing_time': processing_time
        }

    @staticmethod
    def _intervals_to_boundaries(intervals: List[Tuple[float, float]]) -> List[float]:
        if not intervals:
            return []
        return [float(intervals[0][0])] + [float(b) for _, b in intervals]

    @staticmethod
    def _map_box_processed_to_original(
        box_xyxy: Tuple[float, float, float, float],
        scale: float,
        pad_top: int,
        pad_left: int,
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = box_xyxy
        ox1 = (x1 - pad_left) / max(scale, 1e-8)
        oy1 = (y1 - pad_top) / max(scale, 1e-8)
        ox2 = (x2 - pad_left) / max(scale, 1e-8)
        oy2 = (y2 - pad_top) / max(scale, 1e-8)
        return ox1, oy1, ox2, oy2

    def render_visualization(
        self,
        image_bgr: np.ndarray,
        rows: List[Tuple[float, float]],
        cols: List[Tuple[float, float]],
        otsl_sequence: List[str],
        grid_shape: Tuple[int, int],
        *,
        scale: float,
        pad_top: int,
        pad_left: int,
        vis_mode: str = "orig",
        draw_grid: bool = True,
        draw_cells: bool = False,
        draw_merged: bool = True,
        draw_tokens: bool = False,
        line_thickness: Optional[int] = None,
    ) -> np.ndarray:
        """
        将网格/单元格结构可视化叠加到图像上。

        vis_mode:
            - "orig": 画在原图坐标系
            - "processed": 画在 960×960 预处理坐标系（resize+pad 后）
        """
        if vis_mode not in ("orig", "processed"):
            raise ValueError(f"vis_mode 必须为 orig 或 processed，当前: {vis_mode}")

        # 基础画布：用于“抹掉”后续绘制的辅助线时恢复底图像素
        if vis_mode == "processed":
            # 生成与模型输入一致的 960×960 画布（RGB -> BGR）
            processed_rgb, _, _, _ = resize_and_pad(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                self.img_size
            )
            base_canvas = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        else:
            base_canvas = image_bgr

        canvas = base_canvas.copy()

        h, w = canvas.shape[:2]
        if line_thickness is None:
            # 统一使用 1 像素线宽
            line_thickness = 1

        def to_canvas_box(box_xyxy: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
            if vis_mode == "orig":
                box_xyxy = self._map_box_processed_to_original(box_xyxy, scale, pad_top, pad_left)
            x1, y1, x2, y2 = box_xyxy
            x1i = int(round(np.clip(x1, 0, w - 1)))
            y1i = int(round(np.clip(y1, 0, h - 1)))
            x2i = int(round(np.clip(x2, 0, w - 1)))
            y2i = int(round(np.clip(y2, 0, h - 1)))
            if x2i < x1i:
                x1i, x2i = x2i, x1i
            if y2i < y1i:
                y1i, y2i = y2i, y1i
            return x1i, y1i, x2i, y2i

        # 画网格线（由 intervals 的边界构成）
        if draw_grid and rows and cols:
            row_bounds = self._intervals_to_boundaries(rows)
            col_bounds = self._intervals_to_boundaries(cols)

            if len(row_bounds) >= 2 and len(col_bounds) >= 2:
                # 辅助函数：根据坐标模式（原图/预处理图）获取画布上的像素位置
                def get_p(val, is_row=True):
                    if vis_mode == "orig":
                        if is_row:
                            _, oy, _, _ = self._map_box_processed_to_original((0.0, val, 0.0, val), scale, pad_top, pad_left)
                            return int(round(np.clip(oy, 0, h - 1)))
                        else:
                            ox, _, _, _ = self._map_box_processed_to_original((val, 0.0, val, 0.0), scale, pad_top, pad_left)
                            return int(round(np.clip(ox, 0, w - 1)))
                    return int(round(np.clip(val, 0, (h if is_row else w) - 1)))

                # 计算横跨范围：限制在第一线到最后一条线之间
                x_start = get_p(col_bounds[0], is_row=False)
                x_end = get_p(col_bounds[-1], is_row=False) # 最后一条线
                y_start = get_p(row_bounds[0], is_row=True)
                y_end = get_p(row_bounds[-1], is_row=True) # 最后一条线

                # 横线：每条线不超出 x_end
                for y in row_bounds:
                    y_i = get_p(y, is_row=True)
                    cv2.line(canvas, (x_start, y_i), (x_end, y_i), (255, 0, 255), line_thickness)

                # 竖线：每条线不超出 y_end
                for x in col_bounds:
                    x_i = get_p(x, is_row=False)
                    cv2.line(canvas, (x_i, y_start), (x_i, y_end), (255, 0, 255), line_thickness)

        # 画所有原子网格 cell（R*C）框
        if draw_cells and rows and cols:
            for (y1, y2) in rows:
                for (x1, x2) in cols:
                    x1i, y1i, x2i, y2i = to_canvas_box((x1, y1, x2, y2))
                    # cell 框线同样使用紫色，1 像素
                    cv2.rectangle(canvas, (x1i, y1i), (x2i, y2i), (255, 0, 255), line_thickness)

        # 逻辑单元格（合并后的 rowspan/colspan）：
        # - 不再用细线额外画一圈框
        # - 对于合并单元格，直接“抹掉”内部辅助网格线
        # - 默认不在图像上写单元格规模文本
        if draw_merged and rows and cols and otsl_sequence and grid_shape[0] > 0 and grid_shape[1] > 0:
            cells = otsl_sequence_to_spans(otsl_sequence, grid_shape)

            # 预先计算行/列边界，便于定位内部需要抹掉的 grid 线
            row_bounds = self._intervals_to_boundaries(rows)
            col_bounds = self._intervals_to_boundaries(cols)

            for cell in cells:
                rowspan = int(cell["rowspan"])
                colspan = int(cell["colspan"])
                # 对于 1x1 的普通单元格，既不抹线也不打标签
                if rowspan <= 1 and colspan <= 1:
                    continue

                sr, sc = int(cell["row"]), int(cell["col"])
                er = sr + rowspan - 1
                ec = sc + colspan - 1
                if sr < 0 or sc < 0 or er >= len(rows) or ec >= len(cols):
                    continue

                # 该合并单元格在“处理后坐标系”中的整体框
                y1, _ = rows[sr]
                _, y2 = rows[er]
                x1, _ = cols[sc]
                _, x2 = cols[ec]
                x1i, y1i, x2i, y2i = to_canvas_box((x1, y1, x2, y2))

                # 1) 抹掉内部横向 grid 线（行边界）
                for r in range(sr + 1, er + 1):
                    y = row_bounds[r]
                    if vis_mode == "orig":
                        _, oy, _, _ = self._map_box_processed_to_original((0.0, y, 0.0, y), scale, pad_top, pad_left)
                        y_i = int(round(np.clip(oy, 0, h - 1)))
                    else:
                        y_i = int(round(np.clip(y, 0, h - 1)))
                    y_top = max(0, y_i - line_thickness)
                    y_bot = min(h - 1, y_i + line_thickness)
                    canvas[y_top:y_bot + 1, x1i:x2i + 1] = base_canvas[y_top:y_bot + 1, x1i:x2i + 1]

                # 2) 抹掉内部纵向 grid 线（列边界）
                for c in range(sc + 1, ec + 1):
                    x = col_bounds[c]
                    if vis_mode == "orig":
                        ox, _, _, _ = self._map_box_processed_to_original((x, 0.0, x, 0.0), scale, pad_top, pad_left)
                        x_i = int(round(np.clip(ox, 0, w - 1)))
                    else:
                        x_i = int(round(np.clip(x, 0, w - 1)))
                    x_left = max(0, x_i - line_thickness)
                    x_right = min(w - 1, x_i + line_thickness)
                    canvas[y1i:y2i + 1, x_left:x_right + 1] = base_canvas[y1i:y2i + 1, x_left:x_right + 1]

                # 如仍需在合并单元格左上角写规模文本，可将下方代码中
                # draw_tokens 控制逻辑放开使用（当前默认关闭）
                if draw_tokens:
                    label = f"{rowspan}x{colspan}"
                    cv2.putText(
                        canvas,
                        label,
                        (x1i + 2, y1i + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 120, 0),
                        1,
                        cv2.LINE_AA,
                    )

        return canvas

    def _assign_words_to_cells(self,
                                word_bboxes: List[dict],
                                rows: List[Tuple[float, float]],
                                cols: List[Tuple[float, float]],
                                scale: float,
                                pad_top: int,
                                pad_left: int,
                                otsl_sequence: List[str],
                                grid_shape: Tuple[int, int]) -> List[str]:
        """
        将 OCR 文本块分配到对应的逻辑单元格

        对应论文 Section 3.2：
            "based on their positions, the OCR-extracted text blocks are
             sequentially placed into their corresponding table cells"
        """
        from utils.otsl_utils import otsl_sequence_to_spans

        R, C = grid_shape
        cells = otsl_sequence_to_spans(otsl_sequence, grid_shape)

        # 为每个逻辑单元格收集文本
        cell_texts = {(c['row'], c['col']): [] for c in cells}

        for word in word_bboxes:
            bbox = word.get('bbox', [0, 0, 0, 0])
            text = word.get('text', '')
            if not text.strip():
                continue

            # 变换到处理后的图像空间
            cx = ((bbox[0] + bbox[2]) / 2) * scale + pad_left
            cy = ((bbox[1] + bbox[3]) / 2) * scale + pad_top

            # 找到对应的行/列
            row_idx = None
            col_idx = None

            for r, (ymin, ymax) in enumerate(rows):
                if ymin <= cy <= ymax:
                    row_idx = r
                    break

            for c, (xmin, xmax) in enumerate(cols):
                if xmin <= cx <= xmax:
                    col_idx = c
                    break

            if row_idx is None or col_idx is None:
                continue

            # 找到对应的逻辑单元格
            for cell in cells:
                sr, sc = cell['row'], cell['col']
                er = sr + cell['rowspan'] - 1
                ec = sc + cell['colspan'] - 1
                if sr <= row_idx <= er and sc <= col_idx <= ec:
                    cell_texts[(sr, sc)].append(text)
                    break

        # 按单元格顺序生成内容列表
        contents = []
        for cell in cells:
            key = (cell['row'], cell['col'])
            words = cell_texts.get(key, [])
            contents.append(' '.join(words))

        return contents


def parse_args():
    parser = argparse.ArgumentParser(description='TABLET Table Structure Recognition')
    parser.add_argument('--image', type=str, default=None,
                        help='单张表格图像路径')
    parser.add_argument('--data-root', type=str,
                        default=DataConfig.data_root,
                        help='数据集根目录（批量处理模式）')
    parser.add_argument('--split', type=str, default='test',
                        help='数据集分割 (train/val/test)')
    parser.add_argument('--split-checkpoint', '--split-ckpt', type=str,
                        default='checkpoints/split/best.pth',
                        help='Split Model 检查点路径')
    parser.add_argument('--merge-checkpoint', '--merge-ckpt', type=str,
                        default='checkpoints/merge/best.pth',
                        help='Merge Model 检查点路径')
    parser.add_argument('--output-dir', type=str, default='./outputs/predictions',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--img-size', type=int, default=960)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='最大处理样本数（用于调试）')
    parser.add_argument('--use-words', action='store_true',
                        help='使用 words JSON 文件提供 OCR 文本')
    parser.add_argument('--visualize', action='store_true',
                        help='保存推理结果可视化（网格线/合并单元格框）')
    parser.add_argument('--vis-mode', type=str, default='orig', choices=['orig', 'processed'],
                        help='可视化绘制坐标系：orig=原图，processed=960×960预处理图')
    parser.add_argument('--vis-draw-cells', action='store_true',
                        help='额外绘制所有原子网格 cell 框（会更密）')
    parser.add_argument('--vis-draw-tokens', action='store_true',
                        help='在合并单元格左上角绘制跨度标注（rowspan×colspan）')
    parser.add_argument('--vis-ext', type=str, default='jpg', choices=['jpg', 'png'],
                        help='可视化图片输出格式')
    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化推理流水线
    pipeline = TABLETInference(
        split_checkpoint=args.split_checkpoint,
        merge_checkpoint=args.merge_checkpoint,
        device=args.device,
        img_size=args.img_size
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        # 单张图像模式
        result = pipeline.process_single_image(args.image)
        print(f"网格形状: {result['grid_shape']}")
        print(f"OTSL 序列: {result['otsl_sequence'][:20]}...")
        print(f"处理时间: {result['processing_time']:.3f}s")
        out_path = os.path.join(args.output_dir, 'prediction.html')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(result['html'])
        print(f"HTML 已保存到: {out_path}")
        if args.visualize:
            img = cv2.imread(args.image)
            if img is not None:
                vis = pipeline.render_visualization(
                    img,
                    result['rows'],
                    result['cols'],
                    result['otsl_sequence'],
                    result['grid_shape'],
                    scale=result.get('scale', 1.0),
                    pad_top=int(result.get('pad_top', 0)),
                    pad_left=int(result.get('pad_left', 0)),
                    vis_mode=args.vis_mode,
                    draw_grid=True,
                    draw_cells=args.vis_draw_cells,
                    draw_merged=True,
                    draw_tokens=args.vis_draw_tokens,
                )
                vis_path = os.path.join(args.output_dir, f'prediction_vis.{args.vis_ext}')
                cv2.imwrite(vis_path, vis)
                print(f"可视化已保存到: {vis_path}")

    else:
        # 批量处理模式
        images_dir = os.path.join(args.data_root, 'images')
        ann_dir = os.path.join(args.data_root, args.split)
        words_dir = os.path.join(args.data_root, 'words')

        xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.xml')])
        if args.max_samples:
            xml_files = xml_files[:args.max_samples]

        logger.info(f"处理 {len(xml_files)} 个样本...")

        results = {}
        total_time = 0.0

        for i, xml_file in enumerate(xml_files):
            base_name = os.path.splitext(xml_file)[0]
            img_path = os.path.join(images_dir, base_name + '.jpg')

            if not os.path.exists(img_path):
                continue

            # 加载 words（可选）
            word_bboxes = None
            if args.use_words:
                words_path = os.path.join(words_dir, base_name + '_words.json')
                if os.path.exists(words_path):
                    with open(words_path, 'r') as f:
                        word_bboxes = json.load(f)

            try:
                result = pipeline.process_single_image(img_path, word_bboxes)
                total_time += result['processing_time']

                # 保存预测结果
                out_path = os.path.join(args.output_dir, base_name + '.json')
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'html': result['html'],
                        'otsl_sequence': result['otsl_sequence'],
                        'grid_shape': list(result['grid_shape'])
                    }, f, ensure_ascii=False)

                results[base_name] = result['html']

                if args.visualize:
                    img = cv2.imread(img_path)
                    if img is not None:
                        vis = pipeline.render_visualization(
                            img,
                            result['rows'],
                            result['cols'],
                            result['otsl_sequence'],
                            result['grid_shape'],
                            scale=result.get('scale', 1.0),
                            pad_top=int(result.get('pad_top', 0)),
                            pad_left=int(result.get('pad_left', 0)),
                            vis_mode=args.vis_mode,
                            draw_grid=True,
                            draw_cells=args.vis_draw_cells,
                            draw_merged=True,
                            draw_tokens=args.vis_draw_tokens,
                        )
                        vis_path = os.path.join(args.output_dir, f'{base_name}_vis.{args.vis_ext}')
                        cv2.imwrite(vis_path, vis)

                if (i + 1) % 100 == 0:
                    fps = (i + 1) / total_time if total_time > 0 else 0
                    logger.info(f"进度: {i+1}/{len(xml_files)} | FPS: {fps:.2f}")

            except Exception as e:
                logger.error(f"处理失败 {base_name}: {e}")

        avg_fps = len(results) / total_time if total_time > 0 else 0
        logger.info(f"\n处理完成！共 {len(results)} 个样本")
        logger.info(f"平均FPS: {avg_fps:.2f}（论文报告: 18.01 FPS on A100）")


if __name__ == '__main__':
    main()
