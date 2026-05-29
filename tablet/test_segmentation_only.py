import os
import cv2
import numpy as np
import torch
from inference import TABLETInference
from utils.post_process import split_masks_to_grid

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "checkpoints", "split", "best.pth")
MERGE_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "checkpoints", "merge", "best.pth")
IMAGE_PATH = os.environ.get("TABLET_TEST_IMAGE", os.path.join(PROJECT_ROOT, "sample.jpg"))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化推理流水线
    pipeline = TABLETInference(
        split_checkpoint=SPLIT_CKPT,
        merge_checkpoint=MERGE_CKPT,
        device="cuda" if torch.cuda.is_available() else "cpu",
        img_size=960,
    )

    # 读取并预处理
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"无法读取图片: {IMAGE_PATH}")
        return

    # Step 1: 预处理
    image_t, _, scale, pad_top, pad_left = pipeline.preprocess_image(img)

    # Step 2: Split Model → 行/列分割掩码
    row_splits, col_splits = pipeline.run_split(image_t)

    # Step 3: 后处理：分割掩码 → 网格单元 ROI
    rows_all, cols_all, _ = split_masks_to_grid(
        row_splits,
        col_splits,
        img_h=pipeline.img_size,
        img_w=pipeline.img_size,
    )

    # 1. 原始 Split 输出（带外边框）
    vis_all = pipeline.render_visualization(
        img,
        rows_all,
        cols_all,
        otsl_sequence=[],
        grid_shape=(len(rows_all), len(cols_all)),
        scale=scale,
        pad_top=int(pad_top),
        pad_left=int(pad_left),
        vis_mode="orig",
        draw_grid=True,
        draw_cells=False,
        draw_merged=False,
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "seg_with_borders.jpg"), vis_all)

    # 2. 去掉首尾行列（Merge 模型的真实输入范围）
    if len(rows_all) > 2 and len(cols_all) > 2:
        rows_trimmed = rows_all[1:-1]
        cols_trimmed = cols_all[1:-1]
    else:
        rows_trimmed = rows_all
        cols_trimmed = cols_all

    vis_trimmed = pipeline.render_visualization(
        img,
        rows_trimmed,
        cols_trimmed,
        otsl_sequence=[],
        grid_shape=(len(rows_trimmed), len(cols_trimmed)),
        scale=scale,
        pad_top=int(pad_top),
        pad_left=int(pad_left),
        vis_mode="orig",
        draw_grid=True,
        draw_cells=False,
        draw_merged=False,
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "seg_trimmed_for_merge.jpg"), vis_trimmed)

    print(f"处理完成。")
    print(f"1. 带边框（原始Split结果）: outputs/seg_with_borders.jpg")
    print(f"2. 去掉边框（Merge输入范围）: outputs/seg_trimmed_for_merge.jpg")

if __name__ == "__main__":
    main()
