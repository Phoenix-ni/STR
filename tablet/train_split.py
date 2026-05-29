"""
TABLET Split Model 训练脚本

对应论文 Section 4.2 训练配置：
    - 优化器：AdamW (lr=3e-4, betas=(0.9, 0.999), eps=1e-8, wd=5e-4)
    - Batch size：32
    - 梯度裁剪：L2 max_norm=0.5
    - 训练轮数：16 epochs
    - 损失函数：Focal Loss (gamma=2, alpha=1)

使用方法：
    python train_split.py
    python train_split.py --batch-size 16 --epochs 16 --lr 3e-4
    python train_split.py --resume checkpoints/split/best.pth
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
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 禁用 PyTorch nested tensor 原型阶段的警告
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*")

from models.split_model import SplitModel
from losses.focal_loss import SoftTargetFocalLossBinary
from datasets.split_dataset import SplitDataset, split_collate_fn
from configs.config import SplitConfig, TrainConfig, DataConfig

# 日志仅输出 WARNING 以上（进度条由 tqdm 接管终端输出）
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def tqdm_print(msg: str):
    """通过 tqdm 安全打印，避免与进度条混乱"""
    tqdm.write(msg)


def parse_args():
    parser = argparse.ArgumentParser(description='TABLET Split Model Training')
    parser.add_argument('--data-root', type=str,
                        default=DataConfig.data_root,
                        help='数据集根目录')
    parser.add_argument('--batch-size', type=int, default=TrainConfig.batch_size,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=TrainConfig.split_epochs,
                        help='训练轮数（论文：16）')
    parser.add_argument('--lr', type=float, default=TrainConfig.lr,
                        help='初始学习率（论文：3e-4）')
    parser.add_argument('--weight-decay', type=float, default=TrainConfig.weight_decay,
                        help='权重衰减（论文：5e-4）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--num-workers', type=int, default=TrainConfig.num_workers,
                        help='数据加载线程数')
    parser.add_argument('--save-dir', type=str,
                        default=DataConfig.split_checkpoint_dir,
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--img-size', type=int, default=960,
                        help='图像尺寸（论文：960）')
    parser.add_argument('--val-freq', type=int, default=1,
                        help='验证频率（每N个epoch验证一次）')
    parser.add_argument('--log-freq', type=int, default=100,
                        help='日志打印频率（每N个step）')
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='最多使用多少训练样本（用于小样本跑通）')
    parser.add_argument('--max-val-samples', type=int, default=None,
                        help='最多使用多少验证样本（用于小样本跑通）')
    return parser.parse_args()


def compute_split_metrics(row_logits: torch.Tensor,
                           col_logits: torch.Tensor,
                           row_masks: torch.Tensor,
                           col_masks: torch.Tensor) -> dict:
    """
    计算分割任务的评估指标

    返回 F1、精确率、召回率（针对分割位置）
    """
    metrics = {}

    for name, logits, masks in [('row', row_logits, row_masks),
                                  ('col', col_logits, col_masks)]:
        # 取 H/2 分辨率的预测（logits 是 H 的，取奇数索引对应 H/2）
        # 实际上模型输出已经是 H 分辨率（2×上采样后），所以直接比较
        preds = (torch.sigmoid(logits) > 0.5).float()
        targets = masks

        # 转换 masks 到 H 分辨率（masks 是 H/2 的，需要上采样匹配）
        # targets shape: (B, H/2)，preds shape: (B, H)
        # 将 targets 上采样到 H
        targets_full = targets.repeat_interleave(2, dim=1)   # (B, H)

        tp = (preds * targets_full).sum().item()
        fp = (preds * (1 - targets_full)).sum().item()
        fn = ((1 - preds) * targets_full).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[f'{name}_precision'] = precision
        metrics[f'{name}_recall'] = recall
        metrics[f'{name}_f1'] = f1

    return metrics


def train_one_epoch(model: SplitModel,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: SoftTargetFocalLossBinary,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int,
                    log_freq: int = 100) -> dict:
    """训练一个 epoch（含 tqdm 进度条）"""
    model.train()
    total_loss = 0.0
    total_row_loss = 0.0
    total_col_loss = 0.0
    total_row_recall = 0.0
    total_col_recall = 0.0
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

    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        row_masks = batch['row_mask'].to(device, non_blocking=True)
        col_masks = batch['col_mask'].to(device, non_blocking=True)

        row_logits, col_logits = model(images)

        row_masks_full = row_masks.repeat_interleave(2, dim=1)
        col_masks_full = col_masks.repeat_interleave(2, dim=1)

        row_loss = criterion(row_logits, row_masks_full)
        col_loss = criterion(col_logits, col_masks_full)

        # 如果某个分支损失更大（更差），就把它在总损失里“狠狠削弱”
        # 组合方式为：加权平均（权重与另一个分支损失成反比）
        row_det = row_loss.detach()
        col_det = col_loss.detach()
        denom = row_det + col_det + 1e-12
        row_w = col_det / denom
        col_w = row_det / denom
        loss = row_w * row_loss + col_w * col_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
        optimizer.step()

        total_loss += loss.item()
        total_row_loss += row_loss.item()
        total_col_loss += col_loss.item()

        # 计算 step 级别 recall（用于观察 col/row 是否“敢切线”）
        with torch.no_grad():
            row_preds = (torch.sigmoid(row_logits) > 0.5).float()
            col_preds = (torch.sigmoid(col_logits) > 0.5).float()

            row_tp = (row_preds * row_masks_full).sum().item()
            row_fn = ((1.0 - row_preds) * row_masks_full).sum().item()
            row_recall = row_tp / (row_tp + row_fn + 1e-8)

            col_tp = (col_preds * col_masks_full).sum().item()
            col_fn = ((1.0 - col_preds) * col_masks_full).sum().item()
            col_recall = col_tp / (col_tp + col_fn + 1e-8)

        total_row_recall += row_recall
        total_col_recall += col_recall
        n_batches += 1

        avg_loss = total_loss / n_batches
        # 实时更新进度条右侧信息
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{avg_loss:.4f}",
            row=f"{total_row_loss/n_batches:.4f}",
            col=f"{total_col_loss/n_batches:.4f}",
            row_rec=f"{total_row_recall/n_batches:.4f}",
            col_rec=f"{total_col_recall/n_batches:.4f}",
            lr=f"{current_lr:.1e}",
        )

    pbar.close()

    return {
        'loss': total_loss / n_batches,
        'row_loss': total_row_loss / n_batches,
        'col_loss': total_col_loss / n_batches,
        'row_recall': total_row_recall / n_batches,
        'col_recall': total_col_recall / n_batches,
    }


@torch.no_grad()
def validate(model: SplitModel,
             dataloader: DataLoader,
             criterion: SoftTargetFocalLossBinary,
             device: torch.device,
             epoch: int,
             total_epochs: int) -> dict:
    """验证一个 epoch（含 tqdm 进度条）"""
    model.eval()
    total_loss = 0.0
    all_metrics = {'row_f1': [], 'col_f1': [],
                   'row_precision': [], 'col_precision': [],
                   'row_recall': [], 'col_recall': []}
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
        row_masks = batch['row_mask'].to(device)
        col_masks = batch['col_mask'].to(device)

        row_logits, col_logits = model(images)

        row_masks_full = row_masks.repeat_interleave(2, dim=1)
        col_masks_full = col_masks.repeat_interleave(2, dim=1)

        row_loss = criterion(row_logits, row_masks_full)
        col_loss = criterion(col_logits, col_masks_full)

        row_det = row_loss.detach()
        col_det = col_loss.detach()
        denom = row_det + col_det + 1e-12
        row_w = col_det / denom
        col_w = row_det / denom
        loss = row_w * row_loss + col_w * col_loss

        total_loss += loss.item()
        n_batches += 1

        metrics = compute_split_metrics(row_logits, col_logits, row_masks, col_masks)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

        # 实时更新验证进度条
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{total_loss/n_batches:.4f}",
        )

    pbar.close()

    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics['loss'] = total_loss / n_batches
    return avg_metrics


def _plot_split_training_curves(history: dict,
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

    # 统一 x 轴
    x_train = train_epochs
    x_val = val_epochs

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # 1) Loss
    ax = axes[0]
    ax.plot(x_train, history['train_loss'], label='train_loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        ax.plot(x_val, history['val_loss'], label='val_loss', linewidth=2)
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2) F1
    ax = axes[1]
    ax.plot(x_val, history['val_row_f1'], label='val_row_f1', linewidth=2)
    ax.plot(x_val, history['val_col_f1'], label='val_col_f1', linewidth=2)
    ax.set_ylabel('F1')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3) Recall（同时展示 train/val，阈值一致：sigmoid>0.5）
    ax = axes[2]
    ax.plot(x_train, history['train_row_recall'], label='train_row_recall', linewidth=2)
    ax.plot(x_train, history['train_col_recall'], label='train_col_recall', linewidth=2)
    ax.scatter(x_val, history['val_row_recall'], label='val_row_recall', s=18)
    ax.scatter(x_val, history['val_col_recall'], label='val_col_recall', s=18)
    ax.set_ylabel('Recall')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    # 设备配置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    if torch.cuda.device_count() > 1:
        logger.info(f"检测到 {torch.cuda.device_count()} 个 GPU，使用 DataParallel")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # ============================================================
    # 数据加载
    # ============================================================
    train_xml_dir = os.path.join(args.data_root, 'train')
    val_xml_dir = os.path.join(args.data_root, 'val')
    images_dir = os.path.join(args.data_root, 'images')

    logger.info("加载训练数据集...")
    train_dataset = SplitDataset(
        xml_dir=train_xml_dir,
        images_dir=images_dir,
        target_size=args.img_size,
        augment=True
    )

    if args.max_train_samples is not None:
        n = min(int(args.max_train_samples), len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(n)))
        logger.info(f"已限制训练样本数量为前 {n} 个")

    logger.info("加载验证数据集...")
    val_dataset = SplitDataset(
        xml_dir=val_xml_dir,
        images_dir=images_dir,
        target_size=args.img_size,
        augment=False
    )

    if args.max_val_samples is not None:
        n = min(int(args.max_val_samples), len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(n)))
        logger.info(f"已限制验证样本数量为前 {n} 个")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=split_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=split_collate_fn
    )

    logger.info(f"训练集: {len(train_dataset)} 样本，{len(train_loader)} 批次")
    logger.info(f"验证集: {len(val_dataset)} 样本，{len(val_loader)} 批次")

    # ============================================================
    # 模型构建
    # ============================================================
    model = SplitModel(
        img_h=args.img_size,
        img_w=args.img_size,
        fpn_channels=SplitConfig.fpn_out_channels,
        transformer_layers=SplitConfig.transformer_layers,
        transformer_heads=SplitConfig.transformer_heads,
        transformer_ffn_dim=SplitConfig.transformer_ffn_dim,
        transformer_dropout=SplitConfig.transformer_dropout
    )

    # 多GPU
    if torch.cuda.device_count() > 1 and args.device == 'cuda':
        model = nn.DataParallel(model)

    model = model.to(device)

    # 统计参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Split Model 参数量: {n_params/1e6:.1f}M（论文：16.1M）")

    # ============================================================
    # 优化器和损失函数
    # ============================================================
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=TrainConfig.betas,
        eps=TrainConfig.eps,
        weight_decay=args.weight_decay
    )

    # 学习率调度：CosineAnnealing（论文未明确说明，使用常见方案）
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Focal Loss
    # 共享 row/col 的新 loss：软标签容错 + 更强正样本权重
    # dilation_k=1 对应 half-res 约 ±1 bin 容错（更不容易因为对齐偏差而“漏线”）
    criterion = SoftTargetFocalLossBinary(
        gamma=2.0,
        alpha_pos=3.0,
        alpha_neg=1.0,
        dilation_k=1,
        reduction='mean'
    )

    # ============================================================
    # 从检查点恢复
    # ============================================================
    start_epoch = 1
    best_val_f1 = 0.0

    if args.resume and os.path.exists(args.resume):
        logger.info(f"从检查点恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 1) + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        logger.info(f"从 epoch {start_epoch} 继续，最佳 F1: {best_val_f1:.4f}")

    # ============================================================
    # 训练循环
    # ============================================================
    logger.info(f"开始训练 Split Model，共 {args.epochs} 个 epoch")
    logger.info(f"训练配置: lr={args.lr}, bs={args.batch_size}, "
                f"wd={args.weight_decay}, epochs={args.epochs}")

    history = {
        'train_epochs': [],
        'train_loss': [],
        'train_row_loss': [],
        'train_col_loss': [],
        'train_row_recall': [],
        'train_col_recall': [],
        'val_loss': [],
        'val_row_f1': [],
        'val_col_f1': [],
        'val_row_recall': [],
        'val_col_recall': [],
    }
    val_epochs = []

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
        logger.info(f"{'='*60}")

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.epochs, args.log_freq
        )

        logger.info(
            f"Train | Loss: {train_metrics['loss']:.4f} "
            f"(row={train_metrics['row_loss']:.4f}, col={train_metrics['col_loss']:.4f})"
        )

        history['train_epochs'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['train_row_loss'].append(train_metrics['row_loss'])
        history['train_col_loss'].append(train_metrics['col_loss'])
        history['train_row_recall'].append(train_metrics.get('row_recall', 0.0))
        history['train_col_recall'].append(train_metrics.get('col_recall', 0.0))

        # 验证
        if epoch % args.val_freq == 0:
            val_metrics = validate(model, val_loader, criterion, device, epoch, args.epochs)

            logger.info(
                f"Val   | Loss: {val_metrics['loss']:.4f} | "
                f"Row F1: {val_metrics['row_f1']:.4f} | "
                f"Col F1: {val_metrics['col_f1']:.4f}"
            )

            # 保存最佳模型（以行/列 F1 平均值为准）
            val_f1 = (val_metrics['row_f1'] + val_metrics['col_f1']) / 2

            val_epochs.append(epoch)
            history['val_loss'].append(val_metrics['loss'])
            history['val_row_f1'].append(val_metrics['row_f1'])
            history['val_col_f1'].append(val_metrics['col_f1'])
            history['val_row_recall'].append(val_metrics['row_recall'])
            history['val_col_recall'].append(val_metrics['col_recall'])

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_path = os.path.join(args.save_dir, 'best.pth')
                _save_checkpoint(model, optimizer, epoch, best_val_f1,
                                  val_metrics, best_path)
                logger.info(f"✓ 保存最佳模型: F1={best_val_f1:.4f} -> {best_path}")

        # 更新学习率
        scheduler.step()

        # 保存最新检查点
        last_path = os.path.join(args.save_dir, 'last.pth')
        _save_checkpoint(model, optimizer, epoch, best_val_f1,
                          train_metrics, last_path)

    plot_path = os.path.join(args.save_dir, 'split_training_curves.png')
    _plot_split_training_curves(history=history, val_epochs=val_epochs, save_path=plot_path)
    logger.info(f"训练曲线已保存到: {plot_path}")

    logger.info(f"\n训练完成！最佳验证 F1: {best_val_f1:.4f}")


def _save_checkpoint(model, optimizer, epoch, best_val_f1, metrics, path):
    """保存检查点"""
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'metrics': metrics
    }, path)


if __name__ == '__main__':
    main()
