"""
TABLET Merge Model 完整实现

对应论文 Section 3.2：
    1. 标准 ResNet-18 + FPN (256 channels) → P2: H/4 × W/4 × 256
    2. RoIAlign (7×7) 对每个网格单元提取特征
    3. Flatten + 2层 MLP (12544 → 512 → 512)
    4. 展平为序列 R×C → Transformer 编码器（2D位置编码）
    5. Linear(512 → 4) 分类 OTSL 类别 (C, L, U, X)

论文 Section 4.2 参数：
    - FPN: 256 channels
    - RoIAlign: 7×7 输出
    - MLP: 2层 (Linear+ReLU)，input/hidden/output均为512/512
    - Transformer: 3层，8头，FFN=2048，dropout=0.1
    - 2D 可学习位置编码
    - 最大序列长度 640
    - Focal Loss with gamma=2, alpha=1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import List, Tuple, Optional

from models.backbone_merge import BackboneMerge


class TwoLayerMLP(nn.Module):
    """
    2层 MLP：Linear + ReLU + Linear（含 Layer Normalization）

    对应论文 Section 3.2：
        "A two-layer MLP, composed of Linear + ReLU, is used,
         with both output and hidden layers set to 512-dimensional"
    
    改进：添加 Layer Normalization 提高数值稳定性
    """

    def __init__(self, input_dim: int = 12544,
                 hidden_dim: int = 512,
                 output_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        # 权重初始化：Xavier均匀分布
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim)

        Returns:
            (N, output_dim)
        """
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x


class Learnable2DPositionalEmbedding(nn.Module):
    """
    2D 可学习位置编码

    对应论文 Section 3.2：
        "employs a learnable 2-dimensional positional embedding
         to encode spatial relationships effectively"

    实现方式：行位置编码 + 列位置编码之和
    对应 Formerge [35] 中的2D位置编码方案
    """

    def __init__(self, d_model: int = 512, max_rows: int = 50, max_cols: int = 50):
        """
        Args:
            d_model: 特征维度（512）
            max_rows: 最大行数（50）
            max_cols: 最大列数（50）
        """
        super().__init__()
        self.d_model = d_model
        self.max_rows = max_rows
        self.max_cols = max_cols

        # 行位置编码
        self.row_embedding = nn.Embedding(max_rows, d_model)
        # 列位置编码
        self.col_embedding = nn.Embedding(max_cols, d_model)

        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.col_embedding.weight, std=0.02)

    def forward(self, grid_row_ids: torch.Tensor,
                grid_col_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_row_ids: 每个网格单元的行索引 (N,) 或 (B, N)
            grid_col_ids: 每个网格单元的列索引 (N,) 或 (B, N)

        Returns:
            pos_emb: 位置编码 (N, d_model) 或 (B, N, d_model)
        """
        row_emb = self.row_embedding(grid_row_ids)   # (..., d_model)
        col_emb = self.col_embedding(grid_col_ids)   # (..., d_model)
        return row_emb + col_emb                       # (..., d_model)


class TransformerEncoder2D(nn.Module):
    """
    2D Transformer 编码器（用于 Merge Model）

    对应论文 Section 3.2：
        "The feature vector sequence S_grids is then fed into a Transformer encoder"
        参数：3层，8头，FFN=2048，dropout=0.1，2D可学习位置编码
    """

    def __init__(self, d_model: int = 512,
                 num_layers: int = 3,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_rows: int = 50,
                 max_cols: int = 50,
                 max_seq_len: int = 640):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 2D 可学习位置编码
        self.pos_embedding = Learnable2DPositionalEmbedding(
            d_model=d_model, max_rows=max_rows, max_cols=max_cols
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor,
                row_ids: torch.Tensor,
                col_ids: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 序列特征 (B, N, d_model)，N = R×C
            row_ids: 行索引 (B, N)
            col_ids: 列索引 (B, N)
            key_padding_mask: padding mask (B, N)，True表示被mask

        Returns:
            output: (B, N, d_model)
        """
        # 添加 2D 位置编码
        pos_emb = self.pos_embedding(row_ids, col_ids)  # (B, N, d_model)
        x = x + pos_emb

        # Transformer 编码（含 padding mask）
        output = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return output


class MergeModel(nn.Module):
    """
    TABLET Merge Model 完整模型

    功能：对给定网格单元进行 OTSL 分类（C/L/U/X）
    输入：
        images: (B, 3, H, W) 表格图像
        rois_list: List[Tensor] 每张图的网格单元框坐标，每个 (R_i*C_i, 4)，
                   格式为 (x1, y1, x2, y2)，绝对坐标
        grid_shapes: List[Tuple[int, int]] 每张图的网格形状 (R_i, C_i)
    输出：
        logits: (B, max_n, 4) OTSL 分类 logits（已padding）
        padding_mask: (B, max_n) True=padding位置

    对应论文 Section 3.2 及 Section 4.2：
        "The merge model contains 32.5M parameters"
    """

    def __init__(self,
                 img_h: int = 960,
                 img_w: int = 960,
                 fpn_channels: int = 256,
                 roi_output_size: int = 7,
                 mlp_hidden_dim: int = 512,
                 mlp_output_dim: int = 512,
                 transformer_layers: int = 3,
                 transformer_heads: int = 8,
                 transformer_ffn_dim: int = 2048,
                 transformer_dropout: float = 0.1,
                 max_rows: int = 50,
                 max_cols: int = 50,
                 max_seq_len: int = 640,
                 num_classes: int = 4,
                 pretrained_backbone: bool = False):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.fpn_channels = fpn_channels
        self.roi_output_size = roi_output_size
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        # RoIAlign 空间缩放比例（P2层，stride=4）
        self.roi_spatial_scale = 1.0 / 4.0

        # MLP 输入维度：roi_output^2 * fpn_channels
        mlp_input_dim = roi_output_size * roi_output_size * fpn_channels  # 7×7×256=12544

        # 1. 骨干网络：标准 ResNet-18 + FPN
        self.backbone = BackboneMerge(
            fpn_out_channels=fpn_channels,
            pretrained=pretrained_backbone
        )

        # 2. 2层 MLP：12544 → 512 → 512
        self.mlp = TwoLayerMLP(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_output_dim
        )

        # 3. Transformer 编码器（含2D位置编码）
        self.transformer = TransformerEncoder2D(
            d_model=mlp_output_dim,
            num_layers=transformer_layers,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=transformer_dropout,
            max_rows=max_rows,
            max_cols=max_cols,
            max_seq_len=max_seq_len
        )

        # 4. OTSL 分类头：Linear(512 → 4)
        self.classifier = nn.Linear(mlp_output_dim, num_classes)

    def _extract_roi_features(self,
                               feature_map: torch.Tensor,
                               rois_list: List[torch.Tensor],
                               grid_shapes: List[Tuple[int, int]]) -> Tuple[
                                   torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor,
                                   torch.Tensor]:
        """
        使用 RoIAlign 提取每个网格单元的特征

        Args:
            feature_map: P2 特征图 (B, C, H/4, W/4)
            rois_list: 每张图的 ROI 框，List of (N_i, 4)，绝对坐标
            grid_shapes: 每张图的网格形状 [(R_i, C_i), ...]

        Returns:
            cell_feats: (total_cells, d_model) 所有单元格的 MLP 特征
            batch_ids: (total_cells,) 每个单元格属于哪张图
            row_ids_flat: (total_cells,) 行索引
            col_ids_flat: (total_cells,) 列索引
            cell_counts: (B,) 每张图的单元格数量
        """
        B = feature_map.shape[0]
        device = feature_map.device

        all_rois = []
        batch_ids_list = []
        row_ids_list = []
        col_ids_list = []
        cell_counts = []

        for b_idx, (rois, (R, C)) in enumerate(zip(rois_list, grid_shapes)):
            # 使用 rois 的实际长度作为 n_cells（数据集可能已截断）
            n_cells = rois.shape[0]
            if n_cells == 0:
                cell_counts.append(0)
                continue

            # 构建 RoI 格式：(batch_idx, x1, y1, x2, y2)
            batch_col = torch.full((n_cells, 1), b_idx,
                                   dtype=rois.dtype, device=device)
            all_rois.append(torch.cat([batch_col, rois], dim=1))

            batch_ids_list.append(torch.full((n_cells,), b_idx,
                                             dtype=torch.long, device=device))

            # 生成行列索引（按行优先展开）
            # 注意：如果 n_cells < R*C（被截断），索引也需要截断
            row_idx = torch.arange(R, device=device).unsqueeze(1).expand(R, C).reshape(-1)
            col_idx = torch.arange(C, device=device).unsqueeze(0).expand(R, C).reshape(-1)
            row_ids_list.append(row_idx[:n_cells])
            col_ids_list.append(col_idx[:n_cells])
            cell_counts.append(n_cells)

        if not all_rois:
            return (torch.zeros(0, self.max_seq_len, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(B, dtype=torch.long, device=device))

        # 拼接所有 ROI
        all_rois_tensor = torch.cat(all_rois, dim=0)    # (total_cells, 5)

        # RoIAlign 提取特征
        roi_feats = roi_align(
            feature_map,
            all_rois_tensor,
            output_size=self.roi_output_size,
            spatial_scale=self.roi_spatial_scale,
            sampling_ratio=-1,    # 自适应采样
            aligned=True
        )  # (total_cells, C, roi_size, roi_size)

        # Flatten: (total_cells, C * roi_size * roi_size)
        roi_feats_flat = roi_feats.flatten(1)

        # MLP 提取特征
        cell_feats = self.mlp(roi_feats_flat)   # (total_cells, d_model)

        # 合并 batch/row/col 索引
        batch_ids = torch.cat(batch_ids_list, dim=0)
        row_ids_flat = torch.cat(row_ids_list, dim=0)
        col_ids_flat = torch.cat(col_ids_list, dim=0)
        cell_counts_tensor = torch.tensor(cell_counts, dtype=torch.long, device=device)

        return cell_feats, batch_ids, row_ids_flat, col_ids_flat, cell_counts_tensor

    def _pack_sequence(self, cell_feats: torch.Tensor,
                       batch_ids: torch.Tensor,
                       row_ids_flat: torch.Tensor,
                       col_ids_flat: torch.Tensor,
                       cell_counts: torch.Tensor,
                       B: int) -> Tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor, torch.Tensor]:
        """
        将展平的单元格特征打包成 batch 格式，截断到 max_seq_len

        Returns:
            packed_feats: (B, max_n, d_model)
            row_ids_packed: (B, max_n)
            col_ids_packed: (B, max_n)
            padding_mask: (B, max_n) True=padding
        """
        device = cell_feats.device
        d_model = cell_feats.shape[-1]
        max_n = min(int(cell_counts.max().item()), self.max_seq_len)

        packed_feats = torch.zeros(B, max_n, d_model, device=device)
        row_ids_packed = torch.zeros(B, max_n, dtype=torch.long, device=device)
        col_ids_packed = torch.zeros(B, max_n, dtype=torch.long, device=device)
        padding_mask = torch.ones(B, max_n, dtype=torch.bool, device=device)

        offset = 0
        for b_idx in range(B):
            n = cell_counts[b_idx].item()
            if n == 0:
                continue
            n_clip = min(n, self.max_seq_len)
            feats_b = cell_feats[offset: offset + n][:n_clip]
            rows_b = row_ids_flat[offset: offset + n][:n_clip]
            cols_b = col_ids_flat[offset: offset + n][:n_clip]

            packed_feats[b_idx, :n_clip] = feats_b
            row_ids_packed[b_idx, :n_clip] = rows_b
            col_ids_packed[b_idx, :n_clip] = cols_b
            padding_mask[b_idx, :n_clip] = False   # 非padding

            offset += n

        return packed_feats, row_ids_packed, col_ids_packed, padding_mask

    def forward(self,
                images: torch.Tensor,
                rois_list: List[torch.Tensor],
                grid_shapes: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) 表格图像
            rois_list: 每张图的网格单元 ROI，List[Tensor(R*C, 4)]
                       坐标格式：(x1, y1, x2, y2)，绝对坐标（原图尺寸）
            grid_shapes: 每张图的网格形状 [(R_i, C_i), ...]

        Returns:
            logits: (B, max_n, 4) OTSL 分类 logits
            padding_mask: (B, max_n) True=padding
        """
        B = images.shape[0]

        # 1. 提取 P2 特征图
        p2 = self.backbone(images)    # (B, 256, H/4, W/4)

        # 2. RoIAlign + MLP 提取网格单元特征
        cell_feats, batch_ids, row_ids_flat, col_ids_flat, cell_counts = \
            self._extract_roi_features(p2, rois_list, grid_shapes)

        # 3. 打包成 batch 格式
        packed_feats, row_ids_packed, col_ids_packed, padding_mask = \
            self._pack_sequence(cell_feats, batch_ids, row_ids_flat,
                                col_ids_flat, cell_counts, B)

        # 4. Transformer 编码（含2D位置编码）
        transformer_out = self.transformer(
            packed_feats, row_ids_packed, col_ids_packed,
            key_padding_mask=padding_mask
        )   # (B, max_n, 512)

        # 5. OTSL 分类
        logits = self.classifier(transformer_out)   # (B, max_n, 4)

        return logits, padding_mask

    def predict(self,
                images: torch.Tensor,
                rois_list: List[torch.Tensor],
                grid_shapes: List[Tuple[int, int]]) -> List[List[str]]:
        """
        推理模式：返回 OTSL 标签序列

        Returns:
            List of OTSL sequences for each image in batch
        """
        OTSL_ID2LABEL = {0: 'C', 1: 'L', 2: 'U', 3: 'X'}

        with torch.no_grad():
            logits, padding_mask = self.forward(images, rois_list, grid_shapes)
            # logits: (B, max_n, 4)
            pred_ids = logits.argmax(dim=-1)   # (B, max_n)

        results = []
        for b_idx, (R, C) in enumerate(grid_shapes):
            n_cells = R * C
            pred_b = pred_ids[b_idx, :n_cells].cpu().tolist()
            otsl = [OTSL_ID2LABEL[i] for i in pred_b]
            results.append(otsl)
        return results
