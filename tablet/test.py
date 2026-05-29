import os
import random
import argparse
import time

import cv2
import numpy as np
import torch

from inference import TABLETInference
from utils.post_process import split_masks_to_grid

# 配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "fintabnet", "FinTabNet.c-Structure")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "random_vis")

SPLIT_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "checkpoints", "split", "best.pth")
MERGE_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "checkpoints", "merge", "best.pth")
NUM_IMAGES = 50  # 随机处理的图片数量


def parse_args():
    parser = argparse.ArgumentParser(description="TABLET 可视化测试脚本")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="单张图片路径；若不提供则随机抽样多张进行测试",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 根据是否指定 --image 决定处理单张还是随机多张
    img_paths = []
    if args.image is not None:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"指定的图片不存在: {args.image}")
        img_paths = [args.image]
        print(f"单张图片测试: {args.image}")
    else:
        # 列出 images 目录下所有图片
        image_files = [
            f for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_files:
            raise RuntimeError(f"在 {IMAGES_DIR} 下没有找到图片文件")

        # 随机选 NUM_IMAGES 张（若实际少于 NUM_IMAGES 张，则全用）
        if len(image_files) <= NUM_IMAGES:
            selected_files = image_files
        else:
            selected_files = random.sample(image_files, NUM_IMAGES)

        img_paths = [os.path.join(IMAGES_DIR, f) for f in selected_files]
        print(f"将随机处理 {len(img_paths)} 张图片")

    # 初始化推理流水线（建议用 CPU，避免显存 OOM）
    pipeline = TABLETInference(
        split_checkpoint=SPLIT_CKPT,
        merge_checkpoint=MERGE_CKPT,
        device="cuda",       # 如果显存充足可以改成 "cuda"
        img_size=960,
    )

    for idx, img_path in enumerate(img_paths, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{idx}/{len(img_paths)}] 读取失败，跳过: {img_path}")
            continue

        # --- 开始统计 ---
        if pipeline.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        # Step 1: 预处理
        image_t, _processed_rgb, scale, pad_top, pad_left = pipeline.preprocess_image(img)

        # Step 2: Split Model → 行/列分割掩码
        row_splits, col_splits = pipeline.run_split(image_t)

        # Step 3: 后处理：分割掩码 → 网格单元 ROI
        rows, cols, _ = split_masks_to_grid(
            row_splits,
            col_splits,
            img_h=pipeline.img_size,
            img_w=pipeline.img_size,
        )

        # 去掉最外侧一圈“表格外边框线”对应的行/列
        use_rows = rows
        use_cols = cols
        if len(rows) > 2 and len(cols) > 2:
            use_rows = rows[1:-1]
            use_cols = cols[1:-1]

        R = len(use_rows)
        C = len(use_cols)
        if R == 0 or C == 0:
            print(f"[{idx}/{len(img_paths)}] 网格过小或为空，跳过: {img_path}")
            continue

        # 重新构造内部网格的 cell_boxes
        cell_boxes = []
        for r in range(R):
            y1, y2 = use_rows[r]
            for c in range(C):
                x1, x2 = use_cols[c]
                cell_boxes.append([x1, y1, x2, y2])
        cell_boxes = np.array(cell_boxes, dtype=np.float32)

        # Step 4: Merge Model → OTSL 分类（基于去掉外边框后的内部网格）
        grid_shape = (R, C)
        otsl_sequence = pipeline.run_merge(image_t, cell_boxes, grid_shape)

        # --- 结束统计 ---
        if pipeline.device.type == 'cuda':
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        duration = (t_end - t_start) * 1000  # ms
        mem_mb = 0
        if pipeline.device.type == 'cuda':
            mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        print(f"[{idx}/{len(img_paths)}] 处理完成: {img_path}")
        print(f"    - 耗时: {duration:.2f} ms")
        if pipeline.device.type == 'cuda':
            print(f"    - 峰值显存: {mem_mb:.2f} MB")

        vis = pipeline.render_visualization(
            img,
            use_rows,
            use_cols,
            otsl_sequence,
            grid_shape,
            scale=scale,
            pad_top=int(pad_top),
            pad_left=int(pad_left),
            vis_mode="orig",          # 在原图坐标系上绘制
            draw_grid=True,
            draw_cells=False,          # 是否画所有原子 cell
            draw_merged=True,          # 画合并后的单元格（内部网格线会被抹掉）
            draw_tokens=False,         # 不在图像上写单元格规模文本
        )

        # 左边：原始图片；右边：处理后图片
        # 二者尺寸一致，可直接横向拼接
        combined = np.concatenate([img, vis], axis=1)

        base_name, _ = os.path.splitext(os.path.basename(img_path))
        vis_path = os.path.join(OUTPUT_DIR, f"{base_name}_vis.jpg")
        cv2.imwrite(vis_path, combined)

    print(f"全部完成，可视化结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()