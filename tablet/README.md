# TABLET 论文复现

> **论文**：*TABLET: Table Structure Recognition using Encoder-only Transformers*
> **arXiv**：[2506.07015v1](https://arxiv.org/abs/2506.07015)

## 方法概述

TABLET 是一种基于 Split-Merge 的自顶向下表格结构识别方法，利用纯编码器（encoder-only）Transformer 实现高精度、高速度的大规模表格识别。

### 核心创新

1. **行/列分割公式化为序列标注任务**：利用双 Transformer 编码器处理水平/垂直特征序列
2. **单元格合并公式化为网格分类任务**：利用第三个 Transformer 编码器对 RoIAlign 提取的网格特征进行 OTSL 分类
3. **高分辨率特征图**：改进的 ResNet-18（去除 MaxPool + 通道减半）+ FPN，维持 H/2 分辨率

### 系统框架

```
输入表格图像 (960×960)
    │
    ├─ Split Model ─────────────────────────────────────────────────────────
    │   改进版 ResNet-18 + FPN (128ch) → F1/2: 480×480×128
    │   ┌── 行特征提取：全局投影(480×128) + 局部池化(480×240) → 480×368
    │   └── 列特征提取：全局投影(480×128) + 局部池化(480×240) → 480×368
    │   双 Transformer 编码器 (3层,8头,FFN=2048) → 二值分类 → 2×上采样
    │   → 行分割掩码 (960,) + 列分割掩码 (960,)
    │
    ├─ 后处理 ──────────────────────────────────────────────────────────────
    │   分割掩码 → 提取分割线中点 → 生成 R×C 网格 → 单元格 ROI
    │
    └─ Merge Model ─────────────────────────────────────────────────────────
        标准 ResNet-18 + FPN (256ch) → P2: 240×240×256
        RoIAlign (7×7) + 2层MLP(12544→512→512)
        2D Transformer 编码器 (3层,8头,FFN=2048)
        Linear(512→4) → OTSL 分类 (C/L/U/X)
        → HTML 表格结构
```

## 论文实验结果（FinTabNet 测试集）

| 方法 | TEDS (S/C/All) | TEDS-Struc (S/C/All) | Accuracy |
|------|----------------|----------------------|----------|
| TableMaster | - / - / 97.19 | - / - / 98.32 | 82.34 |
| TableFormer | - / - / - | 97.50 / 96.00 / 96.80 | 77.98 |
| **TABLET（本复现目标）** | **98.97 / 98.14 / 98.54** | **99.10 / 98.35 / 98.71** | **88.18** |

S=Simple，C=Complex

## 项目结构

```
TABLET/
├── configs/
│   ├── __init__.py
│   └── config.py              # 全局配置（模型超参、训练配置、路径）
├── datasets/
│   ├── __init__.py
│   ├── xml_parser.py          # FinTabNet XML 标注解析
│   ├── preprocess.py          # 图像预处理（resize+pad）
│   ├── split_dataset.py       # Split Model 数据集
│   └── merge_dataset.py       # Merge Model 数据集
├── models/
│   ├── __init__.py
│   ├── backbone_split.py      # 改进版 ResNet-18 + FPN（Split用）
│   ├── backbone_merge.py      # 标准 ResNet-18 + FPN（Merge用）
│   ├── split_model.py         # TABLET Split Model
│   └── merge_model.py         # TABLET Merge Model
├── losses/
│   ├── __init__.py
│   └── focal_loss.py          # Focal Loss (binary + multiclass)
├── utils/
│   ├── __init__.py
│   ├── post_process.py        # 后处理（分割掩码→网格）
│   ├── otsl_utils.py          # OTSL ↔ HTML 转换
│   └── teds.py                # TEDS/TEDS-Struc/Accuracy 评估
├── data/
│   └── fintabnet/             # FinTabNet 数据集
│       └── FinTabNet.c-Structure/
│           ├── images/        # 表格图像（97,475张）
│           ├── train/         # XML 标注（78,536个）
│           ├── val/           # XML 标注（9,650个）
│           ├── test/          # XML 标注（9,289个）
│           └── words/         # OCR 词框（JSON格式）
├── otsl_labels/               # 预计算 OTSL 标签（97,475个）
│   ├── train/                 # 训练集 OTSL 标签
│   ├── val/                   # 验证集 OTSL 标签
│   └── test/                  # 测试集 OTSL 标签
├── checkpoints/               # 模型检查点
│   ├── split/                 # Split Model 权重
│   └── merge/                 # Merge Model 权重
├── outputs/                   # 推理输出
├── download_data.py           # 数据集下载脚本
├── visualize_annotation.py    # 标注可视化工具
├── train_split.py             # Split Model 训练
├── train_merge.py             # Merge Model 训练
├── inference.py               # 推理流水线
├── evaluate.py                # 评估脚本
└── requirements.txt           # 依赖包
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 数据下载

使用 KaggleHub 一键下载（需要提前在本机配置 Kaggle 账号凭据）：

```bash
python download_data.py
```

下载完成后，数据集默认位于 `data/fintabnet/FinTabNet.c-Structure/`，目录结构与下文「项目结构」一致。

关键统计（FinTabNet 原始划分）：
- 训练集：78,536 张图像（XML 标注 + OTSL 标签）
- 验证集：9,650 张图像
- 测试集：9,289 张图像

### 3. 训练 Split Model

```bash
# 单 GPU 训练
python train_split.py \
    --batch-size 32 \
    --epochs 16 \
    --lr 3e-4 \
    --device cuda

# 多 GPU 训练（自动检测）
python train_split.py --batch-size 32 --epochs 16
```

**论文参数**：batch_size=32, lr=3e-4, AdamW (betas=(0.9,0.999), eps=1e-8, wd=5e-4), 梯度裁剪 max_norm=0.5, 16 epochs

为了便于本地/服务器先跑通流程，可额外使用：
- `--max-train-samples`：限制训练样本数量（取训练集前 N 个）
- `--max-val-samples`：限制验证样本数量（取验证集前 N 个）

训练结束后会自动在 `--save-dir` 下生成 `split_training_curves.png`（包含 loss、`row/col` 的 F1 和 recall 曲线）。

### 4. 训练 Merge Model

```bash
python train_merge.py \
    --batch-size 32 \
    --epochs 24 \
    --lr 3e-4 \
    --device cuda
```

**论文参数**：同 Split Model，另外使用 polynomial LR decay (power=0.9)，24 epochs

注意：Merge Model 训练使用预计算的 OTSL 标签（`otsl_labels/`目录）和 XML 标注（提取网格位置）

训练结束后会自动在 `--save-dir` 下生成 `merge_training_curves.png`（包含 loss、`cell_acc`、`table_acc` 曲线）。

### 5. 推理

```bash
# 单张图像推理
python inference.py \
    --image path/to/table.jpg \
    --split-checkpoint checkpoints/split/best.pth \
    --merge-checkpoint checkpoints/merge/best.pth \
    --output-dir ./outputs/single

# 单张图像 + 可视化输出（掩码 / 网格 / 合并单元格）
# --vis-mode: orig=原图坐标系 | processed=预处理图
# --vis-draw-cells: 绘制所有原子网格 | --vis-draw-tokens: 标注 rowspan×colspan
python inference.py \
    --image path/to/table.jpg \
    --split-checkpoint checkpoints/split/best.pth \
    --merge-checkpoint checkpoints/merge/best.pth \
    --output-dir ./outputs/single_vis \
    --visualize \
    --vis-mode orig \
    --vis-draw-cells \
    --vis-draw-tokens \
    --vis-ext jpg

# 对某个文件夹内所有表格图批量预测（不依赖 FinTabNet 目录结构）
# 会扫描目录下 jpg/png/bmp/webp 等，每张图输出 base_name.json + base_name.html
python inference.py \
    --image-dir /path/to/your/images \
    --split-checkpoint checkpoints/split/best.pth \
    --merge-checkpoint checkpoints/merge/best.pth \
    --output-dir ./outputs/folder_preds \
    --visualize \
    --max-samples 500

# 批量推理（例如对测试集）
# --use-words: 使用 OCR 词框填充单元格（需 words/ 目录）
python inference.py \
    --data-root ./data/fintabnet/FinTabNet.c-Structure \
    --split test \
    --split-checkpoint checkpoints/split/best.pth \
    --merge-checkpoint checkpoints/merge/best.pth \
    --output-dir ./outputs/predictions \
    --use-words \
    --max-samples 1000
```

### 6. 评估

```bash
# 端到端评估（直接推理并评估）
python evaluate.py \
    --end-to-end \
    --split-checkpoint checkpoints/split/best.pth \
    --merge-checkpoint checkpoints/merge/best.pth \
    --split test \
    --n-jobs 4

# 从预测结果评估
python evaluate.py \
    --pred-dir ./outputs/predictions \
    --split test \
    --n-jobs 4
```

## 模型架构详解

### Split Model (16.1M 参数)

```
输入: (B, 3, 960, 960)
│
├─ 改进版 ResNet-18 (去MaxPool, 通道减半: 32/64/128/256)
│  + FPN (128 channels)
│  → F1/2: (B, 128, 480, 480)
│
├─ 行方向特征提取
│  ├─ 全局投影: (B, 128, 480, 480) →[depthwise conv 1×480]→ (B, 480, 128)
│  ├─ 局部特征: (B, 128, 480, 480) →[AvgPool 1×2, 1×1 Conv 128→1]→ (B, 480, 240)
│  └─ 拼接: (B, 480, 368)  [128+240=368]
│
├─ 列方向特征提取 (对称)
│  └─ (B, 480, 368)
│
├─ 双 Transformer 编码器 (各3层, 8头, FFN=2048, dropout=0.1)
│  1D 可学习位置编码（长度480, 维度368）
│
├─ Linear(368→1) + 2×上采样
│
└─ 输出: 行 logits (B, 960) + 列 logits (B, 960)
   损失: `SoftTargetFocalLossBinary`（γ=2, alpha_pos=3, alpha_neg=1, dilation_k=1；row/col 共享同一个损失）
```

### Merge Model (32.5M 参数)

```
输入: (B, 3, 960, 960) + 网格单元 ROI 列表
│
├─ 标准 ResNet-18 + FPN (256 channels)
│  → P2: (B, 256, 240, 240)
│
├─ RoIAlign (7×7, spatial_scale=1/4)
│  对每个网格单元提取 7×7×256=12544 维特征
│
├─ 2层 MLP (12544→512→512)
│
├─ 序列化: (R×C, 512), 最大长度640
│
├─ 2D 可学习位置编码 (行位置嵌入 + 列位置嵌入)
│
├─ Transformer 编码器 (3层, 8头, FFN=2048, dropout=0.1)
│
├─ Linear(512→4)
│
└─ 输出: OTSL 分类 (C/L/U/X) 各网格单元
   损失: Focal Loss (γ=2, α=1)
```

## 数据处理说明

### 现有 OTSL 标签评估

`otsl_labels/` 目录中的 OTSL 标签已完整生成（97,475 个样本），包含：
- `otsl_sequence`: OTSL token 列表（C/L/U/X）
- `grid_shape`: 网格形状 (R, C)
- `cell_spans`: 每个网格位置的跨度信息
- `html`: 对应的 HTML 表格（用于 TEDS 评估）
- `image_size`: 原始图像尺寸

**评估**：现有 OTSL 标签满足 Merge Model 训练需求，可直接使用。Split Model 训练所需的分割掩码由数据集类动态从 XML 标注生成。

### 数据流

- **Split Model 训练**：XML 标注 → 行/列分割线 → H/2 分辨率二值分割 mask
- **Merge Model 训练**：XML 标注（行/列 ROI） + OTSL 标签（token 类别）

## 训练说明

### 硬件要求

- **论文配置**：2× NVIDIA A100 80GB
- **最低配置**：1× GPU with ≥16GB VRAM (batch_size 需相应减小)
- 推荐使用混合精度训练（可在训练脚本中添加 `torch.cuda.amp`）

### 调整 batch_size

论文使用 batch_size=32（2×A100），单 GPU 可减小：

```bash
# 16GB GPU
python train_split.py --batch-size 8   # Split model 较小
python train_merge.py --batch-size 4   # Merge model 含 RoIAlign，内存占用更大
```

### 预期训练时间（单 A100 80GB）

| 模型 | Epochs | 预估时间 |
|------|--------|---------|
| Split | 16 | ~8小时 |
| Merge | 24 | ~20小时 |

## 评估指标说明

论文使用三个指标：

1. **TEDS**：Tree-Edit-Distance Similarity，评估结构+内容（0-100%）
2. **TEDS-Struc**：仅评估结构，不看内容（0-100%）
3. **Accuracy**：完全正确预测的表格比例（TEDS=1.0的样本）

注意：
- TABLET 的 TEDS 和 TEDS-Struc 非常接近（gap<0.2），体现了 Split-Merge 方法的优势：不存在单元格位置与 OCR 文本的错位问题
- 与自回归方法相比，Split-Merge 方法的 Accuracy 显著更高

## 与论文的潜在偏差

复现时可能存在的偏差：
1. **计算资源**：论文使用 2×A100，复现可能需减小 batch_size，影响训练稳定性
2. **数据增强**：论文未详细说明数据增强策略，本复现使用简单的亮度/对比度增强
3. **学习率调度**：Split Model 的 LR 调度论文未明确，本复现使用 CosineAnnealing
4. **OCR 工具**：论文使用专有 OCR 工具，本复现不包含 OCR 模块
5. **FinTabNet 数据质量**：论文提到测试集存在标注错误，复现在标注正确的样本上应能达到更高分数

## 参考

- 论文：Hou & Wang (2025). "TABLET: Table Structure Recognition using Encoder-only Transformers." arXiv:2506.07015
- OTSL：Lysak et al. (2023). "Optimized Table Tokenization for Table Structure Recognition." ICDAR 2023
- FinTabNet：Zheng et al. (2021). "Global Table Extractor (GTE)." WACV 2021
- SPLERGE：Tensmeyer et al. (2019). "Deep Splitting and Merging for Table Structure Decomposition." ICDAR 2019
- Formerge：Nguyen et al. (2023). "Formerge: Recover Spanning Cells in Complex Table Structure." ICDAR 2023
