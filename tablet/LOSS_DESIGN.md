## TABLET 项目中的 Split / Merge 模型损失设计
## 一、Split Model 的损失设计（行 / 列分割）

### 1. 任务形式

- **输入**：表格图像，尺寸固定为 $960 \times 960$。
- **输出**：
  - `row_logits`: 形状 $(B, H)$，$H = 960$，表示每一个水平像素位置是否为**行分割线**的 logit。
  - `col_logits`: 形状 $(B, W)$，$W = 960$，表示每一个垂直像素位置是否为**列分割线**的 logit。

在模型内部（ `models/split_model.py`）：

- 先在下采样后的特征图上（$H/2 \times W/2 = 480 \times 480$）做行 / 列特征提取与 Transformer 编码；
- 得到 $(B, 480)$ 的 `row_logits_half` / `col_logits_half`；
- 通过 `repeat_interleave(2, dim=1)` 做 2× 上采样，得到最终分辨率 $(B, 960)$ 的 `row_logits` / `col_logits`。

### 2. 标签与掩码

在数据集中（`datasets/split_dataset.py`，经由 `split_collate_fn`）：

- `row_mask`: 形状 $(B, H/2)$，二值标签，1 表示该下采样行位置存在行分割线；
- `col_mask`: 形状 $(B, W/2)$，二值标签，1 表示该下采样列位置存在列分割线。

在训练脚本 `train_split.py` 中，为了与模型输出对齐，标签被上采样到原始分辨率：

- `row_masks_full = row_masks.repeat_interleave(2, dim=1)`，形状 $(B, H)$；
- `col_masks_full = col_masks.repeat_interleave(2, dim=1)`，形状 $(B, W)$。

### 3. 损失函数形式

代码中使用 `losses/focal_loss.py` 里的 `SoftTargetFocalLossBinary`，在 `train_split.py` 中的调用为：

- `row_loss = criterion(row_logits, row_masks_full)`
- `col_loss = criterion(col_logits, col_masks_full)`

其中 Split 的总损失使用“反比削弱”动态合并（`train_split.py`）：
$$
w_{row} = \frac{\mathcal{L}_{col}}{\mathcal{L}_{row}+\mathcal{L}_{col}},\quad
w_{col} = \frac{\mathcal{L}_{row}}{\mathcal{L}_{row}+\mathcal{L}_{col}}
$$
$$
\mathcal{L}_{split}
= w_{row}\mathcal{L}_{row} + w_{col}\mathcal{L}_{col}
$$
（实现中对分支 loss 使用 `detach` 后再计算权重）

`SoftTargetFocalLossBinary` 的核心定义包括：

### (1) 软标签（1D dilation / 容错）
给定硬标签 $y \in \{0,1\}$（对应上采样后的 `row_masks_full` / `col_masks_full`），先做 1D dilation 得到软标签 $y^{soft}$：
$$
y^{soft}[i] = \max_{j:\ |j-i|\le k} y[j]
$$
其中 $k$ 是 dilation 半径，对应代码里的 `dilation_k`（默认 $k=1$）。

### (2) focal loss（支持软标签）
令 $p = \sigma(\text{logits})$，soft BCE with logits 定义为：
$$
\text{BCE}(p, y^{soft})
= -\big[y^{soft} \log p + (1-y^{soft})\log (1-p)\big]
$$

并定义：
$$
pt = p \cdot y^{soft} + (1-p)\cdot(1-y^{soft})
$$

正负样本权重（代码中的 `alpha_pos / alpha_neg`）：
$$
\alpha_t = \alpha_{pos}\cdot y^{soft} + \alpha_{neg}\cdot (1-y^{soft})
$$

最终 focal loss：
$$
\text{FL}(p, y^{soft})
  = \alpha_t (1-pt)^\gamma \cdot \text{BCE}(p, y^{soft})
$$

默认超参（当前实现）：
- $\gamma = 2.0$；
- $\alpha_{pos}=3.0,\ \alpha_{neg}=1.0$；
- dilation 半径 $k=1$；
- reduction 使用 `mean`，即对所有位置求平均。

---

## 二、Merge Model 的损失设计（OTSL 单元格合并）

### 1. 任务形式

- **输入**：
  - 表格图像，尺寸 $960 \times 960$；
  - 由 Split 阶段 + 后处理得到的网格单元 ROI 列表 `rois_list`；
  - 每张图的网格形状 `grid_shapes = (R_i, C_i)`。
- **输出**：
  - `logits`: 形状 $(B, \text{max\_n}, 4)$，其中 `max_n` 为当前 batch 内有效单元格数的上界（不超过 `max_seq_len`，默认 640）；
  - 对应的 padding 掩码 `padding_mask`: 形状 $(B, \text{max\_n})$，`True` 表示该位置为 padding。

在模型内部（见 `models/merge_model.py`）：

1. Backbone（ResNet-18 + FPN）输出 P2 特征图；
2. 对每个网格 ROI 通过 `RoIAlign(7×7)` 提取特征并 flatten 成 12544 维；
3. 通过 2 层 MLP 映射到 512 维；
4. 按行优先顺序展平为长度 $N = R \times C$ 的序列，并打包为 batch（截断到 `max_seq_len`，不足位置 padding）；
5. 加上 2D 可学习位置编码后，送入 3 层 Transformer Encoder；
6. 最后 `Linear(512 → 4)` 得到每个格子的 4 类 OTSL logits（C / L / U / X）。

### 2. 标签与 padding 处理

在 `datasets/merge_dataset.py` 中，OTSL 标签被读取并编码为整数序列：

- OTSL 四类标签到 ID 的映射在 `OTSL_LABEL_MAP` 中定义；
- 每个样本的标签按网格顺序展开为一维序列，长度 $R \times C$；
- 在 collate 阶段合并为 batch 时，会对序列做截断 / padding。

在训练脚本 `train_merge.py` 中的关键处理流程：

- 模型前向得到：
  - `logits`: $(B, \text{max\_n}, 4)$
  - `padding_mask`: $(B, \text{max\_n})$，`True` 为 padding 位置；
- 将标签 `otsl_labels` 对齐到同样长度 `max_n`，多则截断，少则在尾部补 `-100`；
- 对于 padding 位置，显式设置 `otsl_labels[padding_mask] = -100`；
- 使用 `ignore_index = -100` 的 Focal Loss，在计算过程中自动忽略这些位置。

### 3. 损失函数形式

代码使用 `losses/focal_loss.py` 中的 `FocalLossMulticlass`，在 `train_merge.py` 中的调用为：

- `loss = criterion(logits, otsl_labels)`

`FocalLossMulticlass` 的核心形式为对交叉熵加上 Focal 权重（在数值稳定实现的基础上）：

1. 对 logits 做 `log_softmax` 得到 `log_probs`；
2. 取目标类别的 log 概率 `target_log_probs`；
3. 从 `log_probs` 恢复概率 $p_k$，并取目标类别概率 `target_probs`；
4. Focal 权重：

$$
\text{focal\_weight}
  = \alpha (1 - p_k)^\gamma
  = \alpha \exp\big(\gamma \log(1 - p_k)\big)
$$

5. Focal Loss：

$$
\text{FL}(p_k) = - \text{focal\_weight} \cdot \log p_k
$$

其中：

- $\gamma = 2.0$；
- $\alpha = 1.0$；
- `ignore_index = -100`：padding 位置完全忽略；
- reduction 为 `mean`，即对所有有效单元格位置求平均。

结合 README 给出的公式，可概括为：

$$
\mathcal{L}_{\text{merge}}
  = \frac{1}{R \times C}
    \sum_{k=1}^{R \times C}
    \alpha_k (1 - p_k)^\gamma \cdot (-\log p_k)
$$

在实现中，$\alpha_k$ 对所有类别统一为 1，$\gamma = 2$，并且只在非 padding 单元格上进行计算。

---

## 三、整体对比与小结

- **Split Model（行 / 列分割）**
  - 任务：对每个像素行 / 列位置做二值分类（是否为分割线）；
  - 损失：**软标签二值 Focal Loss**（对标签做 1D dilation 容错，并使用正负样本权重 $\alpha_{pos}/\alpha_{neg}$）；
  - 合并：对行 / 列分支使用“反比削弱”的动态权重，减弱更差分支对总 loss 的主导效应：
    $$
    \mathcal{L}_{\text{split}}
    = w_{row}\mathcal{L}_{\text{row}} + w_{col}\mathcal{L}_{\text{col}}
    $$
  - 默认超参：$\gamma = 2,\ \alpha_{pos}=3,\ \alpha_{neg}=1,\ k=1$，`mean` reduction。

- **Merge Model（OTSL 单元格合并）**
  - 任务：对每个网格单元做 4 类 OTSL 分类（C / L / U / X）；
  - 损失：**多类别 Focal Loss**，在所有有效单元格上平均，padding 用 `ignore_index` 忽略：
    $$
    \mathcal{L}_{\text{merge}} = \frac{1}{R \times C}\sum_k \alpha (1 - p_k)^\gamma (-\log p_k)
    $$
  - 超参：$\gamma = 2, \alpha = 1$，`mean` reduction，`ignore_index = -100`。

整体来看，项目在 Merge 部分仍沿用了论文风格的多类 focal（padding 用 `ignore_index=-100` 忽略），而 Split 部分为了缓解“漏线/对齐偏差”问题，引入了软标签 dilation 与正负样本权重，并用“反比削弱”动态合并行/列分支的 loss。

