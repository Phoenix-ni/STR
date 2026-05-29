"""
TABLET Split Model 完整实现

对应论文 Section 3.1：
    1. 改进版 ResNet-18 + FPN → F1/2: H/2 × W/2 × 128
    2. 行方向特征提取：
       - 全局投影 FRG: H/2 × 128（沿 W 方向的可学习加权平均）
       - 局部特征 FRL: H/2 × W/4（AvgPool + 1×1 Conv）
       - 拼接 FRG+L: H/2 × (128 + W/4) = 480 × 368
    3. 列方向特征提取（对称）：
       - 全局投影 FCG: W/2 × 128
       - 局部特征 FCL: W/2 × H/4
       - 拼接 FCG+L: W/2 × (128 + H/4) = 480 × 368
    4. 双 Transformer 编码器（3层，8头，FFN=2048）
    5. 线性分类 + 2× 上采样 → H 行二值标签 + W 列二值标签

论文 Section 4.2 参数：
    - 序列长度 480（H/2 = W/2）
    - 特征维度 368（128 + 240 = 128 + W/4）
    - 3层 Transformer，8头，FFN=2048，dropout=0.1
    - 1D 可学习位置编码
    - Focal Loss with gamma=2, alpha=1
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.backbone_split import BackboneSplit


class RowFeatureExtractor(nn.Module):
    """
    行方向特征提取模块

    从 F1/2 (B, C, H/2, W/2) 提取行特征序列 FRG+L (B, H/2, 128+W/4)

    全局特征 FRG: 对宽度方向进行可学习加权平均
    局部特征 FRL: AvgPool(1×2) → 缩减宽度至 W/4，然后1×1 Conv降到1通道
    """

    def __init__(self, in_channels: int = 128, seq_w: int = 480):
        """
        Args:
            in_channels: 输入特征通道数（FPN输出，128）
            seq_w: 水平方向序列长度（W/2 = 480）
        """
        super().__init__()
        self.in_channels = in_channels
        self.seq_w = seq_w

        # 全局投影：可学习加权求和沿 W/2 方向
        # depthwise conv with kernel_size=(1, seq_w) -> (B, C, H/2, 1)
        self.global_proj = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(1, seq_w),
            groups=in_channels
        )  # 输出: (B, C, H/2, 1)

        # 局部特征：AvgPool(1×2) + 1×1 Conv(C→1)
        self.local_pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.local_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, f_half: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_half: (B, C, H/2, W/2) = (B, 128, 480, 480)

        Returns:
            row_features: (B, H/2, 128 + W/4) = (B, 480, 368)
        """
        # 全局特征 FRG: (B, C, H/2, 1) → squeeze → (B, C, H/2) → transpose → (B, H/2, C)
        f_rg = self.global_proj(f_half)             # (B, C, H/2, 1)
        f_rg = f_rg.squeeze(-1).permute(0, 2, 1)    # (B, H/2, C) = (B, 480, 128)

        # 局部特征 FRL: AvgPool→(B, C, H/2, W/4), 1×1 Conv→(B, 1, H/2, W/4)
        f_rl = self.local_pool(f_half)               # (B, C, H/2, W/4) = (B, 128, 480, 240)
        f_rl = self.local_conv(f_rl)                 # (B, 1, H/2, W/4) = (B, 1, 480, 240)
        f_rl = f_rl.squeeze(1)                        # (B, H/2, W/4) = (B, 480, 240)
        # 注意：不需要 permute，squeeze后已经是 (B, H/2, W/4) = (B, 480, 240)

        # 拼接: FRG (B, H/2, 128) + FRL (B, H/2, W/4) = (B, H/2, 128+W/4)
        row_features = torch.cat([f_rg, f_rl], dim=-1)    # (B, 480, 368)
        return row_features


class ColFeatureExtractor(nn.Module):
    """
    列方向特征提取模块

    从 F1/2 (B, C, H/2, W/2) 提取列特征序列 FCG+L (B, W/2, 128+H/4)

    全局特征 FCG: 对高度方向进行可学习加权平均
    局部特征 FCL: AvgPool(2×1) → 缩减高度至 H/4，然后1×1 Conv降到1通道
    """

    def __init__(self, in_channels: int = 128, seq_h: int = 480):
        """
        Args:
            in_channels: 输入特征通道数（FPN输出，128）
            seq_h: 垂直方向序列长度（H/2 = 480）
        """
        super().__init__()
        self.in_channels = in_channels
        self.seq_h = seq_h

        # 全局投影：可学习加权求和沿 H/2 方向
        # depthwise conv with kernel_size=(seq_h, 1) -> (B, C, 1, W/2)
        self.global_proj = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(seq_h, 1),
            groups=in_channels
        )  # 输出: (B, C, 1, W/2)

        # 局部特征：AvgPool(2×1) + 1×1 Conv(C→1)
        self.local_pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.local_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, f_half: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_half: (B, C, H/2, W/2) = (B, 128, 480, 480)

        Returns:
            col_features: (B, W/2, 128 + H/4) = (B, 480, 368)
        """
        # 全局特征 FCG: (B, C, 1, W/2) → squeeze → (B, C, W/2) → transpose → (B, W/2, C)
        f_cg = self.global_proj(f_half)             # (B, C, 1, W/2)
        f_cg = f_cg.squeeze(2).permute(0, 2, 1)    # (B, W/2, C) = (B, 480, 128)

        # 局部特征 FCL: AvgPool→(B, C, H/4, W/2), 1×1 Conv→(B, 1, H/4, W/2)
        f_cl = self.local_pool(f_half)               # (B, C, H/4, W/2) = (B, 128, 240, 480)
        f_cl = self.local_conv(f_cl)                 # (B, 1, H/4, W/2) = (B, 1, 240, 480)
        f_cl = f_cl.squeeze(1).permute(0, 2, 1)     # (B, W/2, H/4) = (B, 480, 240)

        # 拼接: FCG (B, W/2, 128) + FCL (B, W/2, H/4) = (B, W/2, 128+H/4)
        col_features = torch.cat([f_cg, f_cl], dim=-1)    # (B, 480, 368)
        return col_features


class TransformerEncoder1D(nn.Module):
    """
    1D Transformer 编码器（用于 Split Model）

    对应论文 Section 3.1：
        "FRG+L/FCG+L are fed into two Transformer encoders"
        参数：3层，8头，FFN=2048，dropout=0.1，1D可学习位置编码
    """

    def __init__(self, seq_len: int, d_model: int,
                 num_layers: int = 3, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Args:
            seq_len: 序列长度（480）
            d_model: 特征维度（368）
            num_layers: Transformer 层数（3）
            nhead: 注意力头数（8）
            dim_feedforward: FFN 隐层维度（2048）
            dropout: dropout 比率（0.1）
        """
        super().__init__()
        self.d_model = d_model

        # 1D 可学习位置编码（随机初始化并训练），论文 Section 3.1
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.register_buffer('position_ids', torch.arange(seq_len))

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,     # 使用 (B, S, D) 格式
            norm_first=False      # Post-LN（标准Transformer）
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, S, D)，S 为序列长度，D 为特征维度
            key_padding_mask: 可选的 padding mask

        Returns:
            output: (B, S, D)
        """
        B, S, D = x.shape
        # 添加位置编码
        pos_emb = self.pos_embedding(self.position_ids[:S])   # (S, D)
        x = x + pos_emb.unsqueeze(0)                           # (B, S, D)

        # Transformer 编码
        output = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return output


class SplitModel(nn.Module):
    """
    TABLET Split Model 完整模型

    功能：对输入表格图像进行行/列分割预测
    输入：table image (B, 3, H, W)
    输出：
        row_logits: (B, H) 每个水平像素位置的分割 logit
        col_logits: (B, W) 每个垂直像素位置的分割 logit

    对应论文 Section 3.1 及 Section 4.2：
        "The split model contains 16.1M parameters"
        "Input sequence length is 480, each element has dimensionality of 368"
        "3 Transformer layers, 8 attention heads, FFN=2048, dropout=0.1"
    """

    def __init__(self,
                 img_h: int = 960,
                 img_w: int = 960,
                 fpn_channels: int = 128,
                 transformer_layers: int = 3,
                 transformer_heads: int = 8,
                 transformer_ffn_dim: int = 2048,
                 transformer_dropout: float = 0.1):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.feat_h = img_h // 2     # 480
        self.feat_w = img_w // 2     # 480
        self.fpn_channels = fpn_channels  # 128

        # 序列特征维度: 128 + W/4 = 128 + 240 = 368
        self.row_seq_dim = fpn_channels + img_w // 4     # 368
        self.col_seq_dim = fpn_channels + img_h // 4     # 368

        # 1. 骨干网络：改进版 ResNet-18 + FPN
        self.backbone = BackboneSplit(fpn_out_channels=fpn_channels)

        # 2. 行特征提取（global + local projection）
        self.row_extractor = RowFeatureExtractor(
            in_channels=fpn_channels, seq_w=self.feat_w
        )

        # 3. 列特征提取（global + local projection）
        self.col_extractor = ColFeatureExtractor(
            in_channels=fpn_channels, seq_h=self.feat_h
        )

        # 4a. 行 Transformer 编码器（处理水平方向序列）
        self.row_transformer = TransformerEncoder1D(
            seq_len=self.feat_h,         # H/2 = 480
            d_model=self.row_seq_dim,    # 368
            num_layers=transformer_layers,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=transformer_dropout
        )

        # 4b. 列 Transformer 编码器（处理垂直方向序列）
        self.col_transformer = TransformerEncoder1D(
            seq_len=self.feat_w,         # W/2 = 480
            d_model=self.col_seq_dim,    # 368
            num_layers=transformer_layers,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=transformer_dropout
        )

        # 5a. 行分割分类头
        self.row_head = nn.Linear(self.row_seq_dim, 1)

        # 5b. 列分割分类头
        self.col_head = nn.Linear(self.col_seq_dim, 1)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: 输入图像 (B, 3, H, W)，H=W=960，已预处理

        Returns:
            row_logits: (B, H) 行分割 logits，每个像素位置
            col_logits: (B, W) 列分割 logits，每个像素位置
        """
        # 1. 提取 F1/2 特征图
        f_half = self.backbone(images)    # (B, 128, H/2, W/2) = (B, 128, 480, 480)

        # 2. 提取行/列特征序列
        row_seq = self.row_extractor(f_half)   # (B, H/2, 368) = (B, 480, 368)
        col_seq = self.col_extractor(f_half)   # (B, W/2, 368) = (B, 480, 368)

        # 3. Transformer 编码
        row_feats = self.row_transformer(row_seq)   # (B, H/2, 368)
        col_feats = self.col_transformer(col_seq)   # (B, W/2, 368)

        # 4. 分类：每个位置输出一个 logit
        row_logits_half = self.row_head(row_feats).squeeze(-1)   # (B, H/2)
        col_logits_half = self.col_head(col_feats).squeeze(-1)   # (B, W/2)

        # 5. 2× 上采样：每个位置复制一次
        # "each position's classification result is duplicated in place,
        #  effectively performing 2× upsampling" (论文 Section 3.1)
        row_logits = row_logits_half.repeat_interleave(2, dim=1)   # (B, H)
        col_logits = col_logits_half.repeat_interleave(2, dim=1)   # (B, W)

        return row_logits, col_logits

    def predict(self, images: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理模式：返回二值分割结果

        Args:
            images: 输入图像 (B, 3, H, W)
            threshold: 分割阈值

        Returns:
            row_splits: (B, H) 二值掩码，1表示分割位置
            col_splits: (B, W) 二值掩码，1表示分割位置
        """
        with torch.no_grad():
            row_logits, col_logits = self.forward(images)
            row_splits = (torch.sigmoid(row_logits) > threshold).float()
            col_splits = (torch.sigmoid(col_logits) > threshold).float()
        return row_splits, col_splits
