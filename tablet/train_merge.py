"""
TABLET Merge Model 训练脚本

对应论文 Section 4.2 训练配置：
    - 优化器：AdamW (lr=3e-4, betas=(0.9, 0.999), eps=1e-8, wd=5e-4)
    - Batch size：32
    - 梯度裁剪：L2 max_norm=0.5
    - 训练轮数：24 epochs
    - 学习率调度：polynomial decay (power=0.9)
    - 损失函数：Focal Loss (gamma=2, alpha=1)

使用方法：
    python train_merge.py
    python train_merge.py --batch-size 8 --epochs 24
    python train_merge.py --resume checkpoints/merge/best.pth
"""

import os
import sys
import time
import argparse
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# 禁用 PyTorch nested tensor 原型阶段的警告
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*")

from models.merge_model import MergeModel
from losses.focal_loss import FocalLossMulticlass
from datasets.merge_dataset import MergeDataset, merge_collate_fn, OTSL_LABEL_MAP
from configs.config import MergeConfig, TrainConfig, DataConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class PolynomialLRDecay:
    """
    多项式学习率衰减调度器

    对应论文 Section 4.2：
        "the merge model adopts a polynomial decay schedule
         with a decay power of 0.9 for learning rate adjustment"

    公式：lr = lr_base * (1 - t/T)^power
    其中 t 是当前 step，T 是总 step 数
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 max_steps: int, power: float = 0.9, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.power = power
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        t = self.current_step
        T = self.max_steps
        factor = max((1 - t / T) ** self.power, 0.0)

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(base_lr * factor, self.min_lr)

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def parse_args():
    parser = argparse.ArgumentParser(description='TABLET Merge Model Training')
    parser.add_argument('--data-root', type=str,
                        default=DataConfig.data_root)
    parser.add_argument('--otsl-dir', type=str,
                        default=DataConfig.otsl_labels_dir)
    parser.add_argument('--batch-size', type=int, default=TrainConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=TrainConfig.merge_epochs,
                        help='训练轮数（论文：24）')
    parser.add_argument('--lr', type=float, default=TrainConfig.lr)
    parser.add_argument('--weight-decay', type=float, default=TrainConfig.weight_decay)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=TrainConfig.num_workers)
    parser.add_argument('--save-dir', type=str,
                        default=DataConfig.merge_checkpoint_dir)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--img-size', type=int, default=960)
    parser.add_argument('--max-seq-len', type=int,
                        default=MergeConfig.max_seq_len,
                        help='最大序列长度（论文：640）')
    parser.add_argument('--val-freq', type=int, default=1)
    parser.add_argument('--log-freq', type=int, default=100)
    parser.add_argument('--pretrained-backbone', action='store_true',
                        help='使用 ImageNet 预训练 ResNet-18 骨干网络')
    return parser.parse_args()


def compute_merge_metrics(logits: torch.Tensor,
                           labels: torch.Tensor,
                           padding_mask: torch.Tensor,
                           num_cells_list: list) -> dict:
    """
    计算 Merge 任务的评估指标

    Args:
        logits: (B, max_n, 4) 预测 logits
        labels: (B, max_n) 真实标签（-100=padding）
        padding_mask: (B, max_n) True=padding
        num_cells_list: 每个样本的有效单元格数

    Returns:
        metrics dict
    """
    pred_ids = logits.argmax(dim=-1)   # (B, max_n)

    # 计算 per-cell 准确率（忽略 padding）
    valid_mask = ~padding_mask                     # True=valid
    valid_preds = pred_ids[valid_mask]
    valid_labels = labels[valid_mask]

    # 过滤 ignore_index=-100
    valid_idx = valid_labels != -100
    valid_preds = valid_preds[valid_idx]
    valid_labels = valid_labels[valid_idx]

    cell_acc = (valid_preds == valid_labels).float().mean().item()

    # 计算完整表格准确率
    B = logits.shape[0]
    correct_tables = 0
    for b in range(B):
        n = num_cells_list[b]
        pred_b = pred_ids[b, :n]
        label_b = labels[b, :n]
        if (pred_b == label_b).all():
            correct_tables += 1
    table_acc = correct_tables / B

    # 各类别 F1（OTSL C/L/U/X）
    class_names = ['C', 'L', 'U', 'X']
    per_class_f1 = {}
    for c in range(4):
        tp = ((valid_preds == c) & (valid_labels == c)).sum().item()
        fp = ((valid_preds == c) & (valid_labels != c)).sum().item()
        fn = ((valid_preds != c) & (valid_labels == c)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        per_class_f1[f'{class_names[c]}_f1'] = f1

    return {
        'cell_acc': cell_acc,
        'table_acc': table_acc,
        **per_class_f1
    }


def train_one_epoch(model: MergeModel,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: FocalLossMulticlass,
                    scheduler,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int,
                    log_freq: int = 100) -> dict:
    """训练一个 epoch（含 tqdm 进度条）"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    current_lr = optimizer.param_groups[0]['lr']

    # ── 步级进度条 ──────────────────────────────────────────────
    pbar = tqdm(
        dataloader,
        desc=f"  Train [{epoch:>2}/{total_epochs}]",
        unit="batch",
        ncols=110,
        leave=True,
        colour='cyan'
    )

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        rois_list = [r.to(device) for r in batch['rois_list']]
        otsl_labels = batch['otsl_labels'].to(device)
        grid_shapes = batch['grid_shapes']
        num_cells = batch['num_cells']

        # 前向传播
        logits, padding_mask = model(images, rois_list, grid_shapes)
        # logits: (B, max_n, 4)

        # 计算 Focal Loss（忽略 padding 位置）
        B, max_n, num_classes = logits.shape

        # 对齐 otsl_labels 的长度到 max_n
        if otsl_labels.shape[1] > max_n:
            otsl_labels = otsl_labels[:, :max_n]
        elif otsl_labels.shape[1] < max_n:
            pad = torch.full((B, max_n - otsl_labels.shape[1]),
                             -100, dtype=torch.long, device=device)
            otsl_labels = torch.cat([otsl_labels, pad], dim=1)

        # 设置 padding 位置的标签为 -100（忽略）
        otsl_labels[padding_mask] = -100

        loss = criterion(logits, otsl_labels)

        # 反向传播（含数值稳定性检查）
        optimizer.zero_grad()
        loss.backward()
        
        # 检查 NaN/Inf
        has_nan = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan = True
                break
        
        if has_nan:
            logger.warning(f"Batch {batch_idx}: 检测到 NaN/Inf 梯度，跳过此步")
            optimizer.zero_grad()
            continue
        
        # 梯度裁剪（更激进以防止爆炸）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2)
        optimizer.step()

        # Polynomial LR decay 按 step 更新
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        avg_loss = total_loss / n_batches
        current_lr = optimizer.param_groups[0]['lr']
        # 实时更新进度条右侧信息
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{avg_loss:.4f}",
            lr=f"{current_lr:.1e}",
        )

    pbar.close()

    return {'loss': total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(model: MergeModel,
             dataloader: DataLoader,
             criterion: FocalLossMulticlass,
             device: torch.device,
             epoch: int,
             total_epochs: int) -> dict:
    """验证一个 epoch（含 tqdm 进度条）"""
    model.eval()
    total_loss = 0.0
    all_cell_accs = []
    all_table_accs = []
    n_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"  Val   [{epoch:>2}/{total_epochs}]",
        unit="batch",
        ncols=110,
        leave=True,
        colour='green'
    )

    for batch in pbar:
        images = batch['image'].to(device)
        rois_list = [r.to(device) for r in batch['rois_list']]
        otsl_labels = batch['otsl_labels'].to(device)
        grid_shapes = batch['grid_shapes']
        num_cells = batch['num_cells']

        logits, padding_mask = model(images, rois_list, grid_shapes)

        B, max_n, _ = logits.shape
        if otsl_labels.shape[1] > max_n:
            otsl_labels = otsl_labels[:, :max_n]
        elif otsl_labels.shape[1] < max_n:
            pad = torch.full((B, max_n - otsl_labels.shape[1]),
                             -100, dtype=torch.long, device=device)
            otsl_labels = torch.cat([otsl_labels, pad], dim=1)

        otsl_labels[padding_mask] = -100

        loss = criterion(logits, otsl_labels)
        total_loss += loss.item()
        n_batches += 1

        metrics = compute_merge_metrics(logits, otsl_labels, padding_mask, num_cells)
        all_cell_accs.append(metrics['cell_acc'])
        all_table_accs.append(metrics['table_acc'])

        # 实时更新验证进度条
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_loss/n_batches:.4f}",
        )

    pbar.close()

    return {
        'loss': total_loss / max(n_batches, 1),
        'cell_acc': np.mean(all_cell_accs),
        'table_acc': np.mean(all_table_accs),
    }


def _plot_merge_training_curves(history: dict,
                                 val_epochs: list,
                                 save_path: str) -> None:
    """
    训练结束后绘制曲线（headless 服务器：使用 Agg 后端）
    """
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"未安装/无法使用 matplotlib，跳过绘图: {e}")
        return

    train_epochs = history.get('train_epochs', [])
    if not train_epochs:
        return

    x_train = train_epochs
    x_val = val_epochs

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 1) Loss
    ax = axes[0]
    ax.plot(x_train, history['train_loss'], label='train_loss', linewidth=2)
    if history.get('val_loss'):
        ax.plot(x_val, history['val_loss'], label='val_loss', linewidth=2)
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) Acc
    ax = axes[1]
    if history.get('val_cell_acc'):
        ax.plot(x_val, history['val_cell_acc'], label='val_cell_acc', linewidth=2)
    if history.get('val_table_acc'):
        ax.plot(x_val, history['val_table_acc'], label='val_table_acc', linewidth=2)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ============================================================
    # 数据加载
    # ============================================================
    train_xml_dir = os.path.join(args.data_root, 'train')
    val_xml_dir = os.path.join(args.data_root, 'val')
    images_dir = os.path.join(args.data_root, 'images')
    train_otsl_dir = os.path.join(args.otsl_dir, 'train')
    val_otsl_dir = os.path.join(args.otsl_dir, 'val')

    logger.info("加载 Merge 训练数据集...")
    train_dataset = MergeDataset(
        xml_dir=train_xml_dir,
        otsl_dir=train_otsl_dir,
        images_dir=images_dir,
        target_size=args.img_size,
        max_seq_len=args.max_seq_len,
        augment=True
    )

    logger.info("加载 Merge 验证数据集...")
    val_dataset = MergeDataset(
        xml_dir=val_xml_dir,
        otsl_dir=val_otsl_dir,
        images_dir=images_dir,
        target_size=args.img_size,
        max_seq_len=args.max_seq_len,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=merge_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=merge_collate_fn
    )

    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")

    # ============================================================
    # 模型构建
    # ============================================================
    model = MergeModel(
        img_h=args.img_size,
        img_w=args.img_size,
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
        max_seq_len=args.max_seq_len,
        num_classes=MergeConfig.num_classes,
        pretrained_backbone=args.pretrained_backbone
    )

    if torch.cuda.device_count() > 1 and args.device == 'cuda':
        model = nn.DataParallel(model)

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Merge Model 参数量: {n_params/1e6:.1f}M（论文：32.5M）")

    # ============================================================
    # 优化器和学习率调度
    # ============================================================
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=TrainConfig.betas,
        eps=TrainConfig.eps,
        weight_decay=args.weight_decay
    )

    # Polynomial Decay 按总 step 计算
    total_steps = len(train_loader) * args.epochs
    scheduler = PolynomialLRDecay(
        optimizer,
        max_steps=total_steps,
        power=TrainConfig.merge_lr_decay_power   # 0.9
    )

    # Focal Loss
    criterion = FocalLossMulticlass(
        gamma=2.0, alpha=1.0,
        num_classes=MergeConfig.num_classes,
        reduction='mean',
        ignore_index=-100
    )

    # ============================================================
    # 从检查点恢复
    # ============================================================
    start_epoch = 1
    best_val_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        logger.info(f"从检查点恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 1) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        scheduler.current_step = ckpt.get('scheduler_step', 0)
        logger.info(f"从 epoch {start_epoch} 继续，最佳验证准确率: {best_val_acc:.4f}")

    # ============================================================
    # 训练循环
    # ============================================================
    logger.info(f"开始训练 Merge Model，共 {args.epochs} 个 epoch")
    logger.info(f"训练配置: lr={args.lr}, bs={args.batch_size}, "
                f"polynomial_power={TrainConfig.merge_lr_decay_power}")

    history = {
        'train_epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_cell_acc': [],
        'val_table_acc': [],
    }
    val_epochs = []

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.2e}")
        logger.info(f"{'='*60}")

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scheduler, device, epoch, args.epochs, args.log_freq
        )

        logger.info(f"Train | Loss: {train_metrics['loss']:.4f}")

        history['train_epochs'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])

        # 验证
        if epoch % args.val_freq == 0:
            val_metrics = validate(model, val_loader, criterion, device, epoch, args.epochs)

            logger.info(
                f"Val   | Loss: {val_metrics['loss']:.4f} | "
                f"Cell Acc: {val_metrics['cell_acc']:.4f} | "
                f"Table Acc: {val_metrics['table_acc']:.4f}"
            )

            val_acc = val_metrics['cell_acc']

            val_epochs.append(epoch)
            history['val_loss'].append(val_metrics['loss'])
            history['val_cell_acc'].append(val_metrics['cell_acc'])
            history['val_table_acc'].append(val_metrics['table_acc'])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(args.save_dir, 'best.pth')
                _save_checkpoint(model, optimizer, epoch, best_val_acc,
                                  val_metrics, scheduler.current_step, best_path)
                logger.info(f"✓ 保存最佳模型: Acc={best_val_acc:.4f} -> {best_path}")

        # 保存最新检查点
        last_path = os.path.join(args.save_dir, 'last.pth')
        _save_checkpoint(model, optimizer, epoch, best_val_acc,
                          train_metrics, scheduler.current_step, last_path)

    plot_path = os.path.join(args.save_dir, 'merge_training_curves.png')
    _plot_merge_training_curves(history=history, val_epochs=val_epochs, save_path=plot_path)
    logger.info(f"训练曲线已保存到: {plot_path}")

    logger.info(f"\n训练完成！最佳验证单元格准确率: {best_val_acc:.4f}")


def _save_checkpoint(model, optimizer, epoch, best_val_acc, metrics,
                      scheduler_step, path):
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'scheduler_step': scheduler_step,
        'metrics': metrics
    }, path)


if __name__ == '__main__':
    main()
