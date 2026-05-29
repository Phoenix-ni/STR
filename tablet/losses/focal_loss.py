"""
Focal Loss 实现
对应论文 Section 3.1 & 3.2 中的损失函数

Split Model:
    FL_split = 1/n_h * sum(alpha_i * (1-p_i)^gamma * (-log(p_i)))
             + 1/n_v * sum(alpha_j * (1-p_j)^gamma * (-log(p_j)))

Merge Model:
    FL_merge = 1/(R*C) * sum(alpha_k * (1-p_k)^gamma * (-log(p_k)))

论文参数：alpha=1, gamma=2，对所有类别平等对待
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossBinary(nn.Module):
    """
    二值 Focal Loss，用于 Split Model 的行/列分割任务

    对应论文 Section 3.1：
        FL_split = 1/n * sum((1-p)^gamma * (-log(p)))
        其中 n 是序列长度（H/2 或 W/2），alpha=1, gamma=2
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            gamma: 聚焦参数，论文中为2
            alpha: 类别权重，论文中为1（不做平衡）
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 模型输出的 logits，shape (B, N) 或 (B, N, 1)
            targets: 真实标签（0或1），shape (B, N) 或 (B, N, 1)

        Returns:
            focal loss 值
        """
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        if targets.dim() == 3:
            targets = targets.squeeze(-1)

        # BCE loss with logits: -[y*log(p) + (1-y)*log(1-p)]
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )

        # 计算预测概率 p
        probs = torch.sigmoid(logits)

        # Focal 权重：对于正样本 alpha*(1-p)^gamma，对于负样本 alpha*p^gamma
        # 由于 alpha=1，两者统一为 (1-pt)^gamma
        # pt = p if target==1 else (1-p)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SoftTargetFocalLossBinary(nn.Module):
    """
    SoftTarget Focal Loss（用于 Split 的二值行/列分割）

    设计要点：
    1) 对硬 0/1 标签做 1D dilation，把真实线附近的一小段都当作正样本区域（提升对齐偏差鲁棒性）
    2) 使用 alpha_pos/alpha_neg 对正负样本进行加权，避免模型过于保守导致“漏线”
    3) 仍使用 focal 的 (1-pt)^gamma 进行难样本聚焦
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha_pos: float = 3.0,
                 alpha_neg: float = 1.0,
                 dilation_k: int = 1,
                 reduction: str = 'mean'):
        """
        Args:
            gamma: focal 聚焦参数
            alpha_pos: 正样本权重
            alpha_neg: 负样本权重
            dilation_k: 半径（在 half-res 的 1D 序列上对正样本做容错扩张），k=1 => ±1 bin
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.dilation_k = int(dilation_k)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, N) 或 (B, N, 1)
            targets: (B, N) 或 (B, N, 1) 的 0/1 标签（float 或 int）
        """
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        if targets.dim() == 3:
            targets = targets.squeeze(-1)

        targets = targets.to(dtype=logits.dtype)

        # 1D dilation：把 1 的附近区间也视作“软正样本区域”
        if self.dilation_k > 0:
            # targets: (B, N) -> (B, 1, N)
            t = targets.unsqueeze(1)
            k = 2 * self.dilation_k + 1
            # max_pool1d 相当于 1D 的“邻域存在即为 1”
            t_dilated = F.max_pool1d(
                t,
                kernel_size=k,
                stride=1,
                padding=self.dilation_k
            )
            soft_targets = t_dilated.squeeze(1)
        else:
            soft_targets = targets

        # BCE with logits（支持软标签）
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, soft_targets, reduction='none'
        )

        probs = torch.sigmoid(logits)
        # pt 的定义对软标签同样成立：pt = P(y=1)*p + P(y=0)*(1-p)
        pt = probs * soft_targets + (1.0 - probs) * (1.0 - soft_targets)

        alpha_t = self.alpha_pos * soft_targets + self.alpha_neg * (1.0 - soft_targets)
        focal_weight = alpha_t * (1.0 - pt).pow(self.gamma)

        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossMulticlass(nn.Module):
    """
    多类别 Focal Loss，用于 Merge Model 的 OTSL 分类任务

    对应论文 Section 3.2：
        FL_merge = 1/(R*C) * sum(alpha_k * (1-p_k)^gamma * (-log(p_k)))
        其中 alpha_k=1, gamma=2
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0,
                 num_classes: int = 4, reduction: str = 'mean',
                 ignore_index: int = -100):
        """
        Args:
            gamma: 聚焦参数，论文中为2
            alpha: 类别权重，论文中为1
            num_classes: OTSL 类别数（C, L, U, X = 4类）
            reduction: 'mean' | 'sum' | 'none'
            ignore_index: 忽略索引（用于padding）
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 模型输出，shape (N, num_classes) 或 (B, N, num_classes)
            targets: 整数类别标签，shape (N,) 或 (B, N)

        Returns:
            focal loss 值 (数值稳定版本)
        """
        # 展平到2D
        if logits.dim() == 3:
            B, N, C = logits.shape
            logits = logits.reshape(B * N, C)
            targets = targets.reshape(B * N)

        # 过滤掉 ignore_index
        valid_mask = targets != self.ignore_index
        logits = logits[valid_mask]
        targets = targets[valid_mask]

        if logits.numel() == 0:
            return logits.sum() * 0.0

        # ============================================================
        # 数值稳定的 Focal Loss 实现
        # 使用 log-space 计算避免数值溢出/下溢
        # ============================================================
        
        # 计算 softmax 概率（使用 log 域）
        log_probs = F.log_softmax(logits, dim=-1)   # (N, C)
        
        # 取目标类别的 log 概率
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        
        # 直接从 log_probs 恢复概率（更稳定）
        # 避免 exp(log_probs) 导致的数值问题
        probs = torch.clamp(torch.exp(log_probs), min=1e-7, max=1.0 - 1e-7)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        
        # 数值稳定的 focal weight 计算
        # (1-p)^gamma = exp(gamma * log(1-p))
        safe_p = torch.clamp(target_probs, min=1e-7, max=1.0 - 1e-7)
        log_focal_weight = self.gamma * torch.log(1.0 - safe_p)
        focal_weight = self.alpha * torch.exp(log_focal_weight)
        
        # 为了进一步的数值稳定性，限制 focal_weight 的范围
        focal_weight = torch.clamp(focal_weight, min=0.0, max=1e3)
        
        # Focal loss: -focal_weight * log(p_k)
        # 使用更稳定的计算方式
        focal_loss = -focal_weight * target_log_probs
        
        # 检查并处理 NaN/Inf
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            # 如果仍有数值问题，使用标准交叉熵的降级方案
            focal_loss = F.cross_entropy(logits, targets, reduction='none')
        
        if self.reduction == 'mean':
            loss_mean = focal_loss.mean()
            # 最终检查
            if torch.isnan(loss_mean) or torch.isinf(loss_mean):
                return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            return loss_mean
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
