"""
TABLET 论文复现配置文件
对应论文 Section 4.2: Implementation Details
"""

# ============================================================
# 图像尺寸（论文 Section 4.1）
# ============================================================
IMG_H = 960
IMG_W = 960

# ============================================================
# Split Model 配置（论文 Section 3.1 & 4.2）
# ============================================================
class SplitConfig:
    # 骨干网络：去除MaxPool、减半通道数的 ResNet-18
    backbone_channels_half = True      # 减半通道：32, 64, 128, 256
    backbone_remove_maxpool = True     # 去除初始MaxPool
    fpn_out_channels = 128             # FPN输出通道数

    # Feature map 尺寸（H=W=960）
    # F1/2: H/2 × W/2 × 128 = 480 × 480 × 128
    feat_h = IMG_H // 2                # 480
    feat_w = IMG_W // 2                # 480

    # 行/列序列特征维度（论文中为368）
    # Row seq dim = 128 + W/4 = 128 + 240 = 368
    # Col seq dim = 128 + H/4 = 128 + 240 = 368
    row_seq_dim = fpn_out_channels + IMG_W // 4   # 128 + 240 = 368
    col_seq_dim = fpn_out_channels + IMG_H // 4   # 128 + 240 = 368

    # Transformer 编码器配置（论文 Section 4.2）
    transformer_layers = 3
    transformer_heads = 8
    transformer_ffn_dim = 2048
    transformer_dropout = 0.1

    # 输出序列长度
    row_seq_len = IMG_H // 2           # 480（水平方向）
    col_seq_len = IMG_W // 2           # 480（垂直方向）

    # 分割区域最小宽度（像素），论文 Section 4.1
    min_split_width = 5

    # 参数量（论文说16.1M）
    num_params_approx = 16_100_000


# ============================================================
# Merge Model 配置（论文 Section 3.2 & 4.2）
# ============================================================
class MergeConfig:
    # 骨干网络：标准 ResNet-18
    backbone_channels_half = False
    backbone_remove_maxpool = False
    fpn_out_channels = 256             # FPN输出通道数

    # RoIAlign 配置
    roi_output_size = 7                # 7×7 输出
    roi_spatial_scale = 1 / 4         # P2层，步长4（H/4 × W/4）

    # MLP 配置（Linear + ReLU，2层）
    # 输入：7×7×256 = 12544
    mlp_input_dim = roi_output_size * roi_output_size * fpn_out_channels  # 12544
    mlp_hidden_dim = 512
    mlp_output_dim = 512              # Transformer 输入维度

    # Transformer 编码器配置
    transformer_layers = 3
    transformer_heads = 8
    transformer_ffn_dim = 2048
    transformer_dropout = 0.1
    transformer_d_model = 512

    # 序列最大长度（R×C ≤ 640）
    max_seq_len = 640

    # 2D 位置编码的最大行/列数
    # 注：max_seq_len=640 允许 R 或 C 达到 640，故需足够大的值避免索引越界
    max_rows = 640
    max_cols = 640

    # OTSL 标签类别（C, L, U, X），不含NL
    num_classes = 4
    OTSL_C = 0    # 新单元格
    OTSL_L = 1    # 左合并
    OTSL_U = 2    # 上合并
    OTSL_X = 3    # 交叉合并

    OTSL_LABEL_MAP = {'C': 0, 'L': 1, 'U': 2, 'X': 3}
    OTSL_ID2LABEL = {0: 'C', 1: 'L', 2: 'U', 3: 'X'}

    # 参数量（论文说32.5M）
    num_params_approx = 32_500_000


# ============================================================
# Focal Loss 配置（论文 Section 3.1 & 3.2）
# ============================================================
class FocalLossConfig:
    gamma = 2.0         # 聚焦参数
    alpha = 1.0         # 类别权重（全部设为1）


# ============================================================
# 训练配置（论文 Section 4.2）
# ============================================================
class TrainConfig:
    # 优化器：AdamW
    optimizer = 'AdamW'
    lr = 3e-4  # 论文 Section 4.2：原始学习率值
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 5e-4

    # 梯度裁剪
    grad_clip_max_norm = 0.5  # 论文 Section 4.2：L2 norm 梯度裁剪
    grad_clip_norm_type = 2.0         # L2 normalization

    # 批次大小：论文 Section 4.2（2×A100 GPU 上的配置）
    batch_size = 32

    # 各模型训练轮数
    split_epochs = 16
    merge_epochs = 24

    # Merge Model 学习率调度：polynomial decay, power=0.9
    merge_lr_decay_power = 0.9

    # 数据加载
    num_workers = 8
    pin_memory = True


# ============================================================
# 数据集路径配置
# ============================================================
class DataConfig:
    # 数据根目录
    data_root = './data/fintabnet/FinTabNet.c-Structure'
    images_dir = f'{data_root}/images'
    words_dir = f'{data_root}/words'

    # 各split的XML标注目录
    train_ann_dir = f'{data_root}/train'
    val_ann_dir = f'{data_root}/val'
    test_ann_dir = f'{data_root}/test'

    # OTSL 标签目录
    otsl_labels_dir = './otsl_labels'
    otsl_train_dir = f'{otsl_labels_dir}/train'
    otsl_val_dir = f'{otsl_labels_dir}/val'
    otsl_test_dir = f'{otsl_labels_dir}/test'

    # 模型保存目录
    checkpoints_dir = './checkpoints'
    split_checkpoint_dir = f'{checkpoints_dir}/split'
    merge_checkpoint_dir = f'{checkpoints_dir}/merge'

    # 输出目录
    outputs_dir = './outputs'
