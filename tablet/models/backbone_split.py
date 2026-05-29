"""
Split Model 骨干网络：改进版 ResNet-18 + FPN

对应论文 Section 3.1：
    - 去除 ResNet-18 的 MaxPool 层（保持 H/2 分辨率）
    - 将 ResNet-18 通道数减半（32, 64, 128, 256）
    - FPN 输出通道数为 128
    - 最终输出 F1/2: H/2 × W/2 × 128

论文说明：
    "we remove the Max Pooling layer from ResNet and halve the number of channels
     to reduce computational complexity, and combine it with a FPN having 128 channels.
     The resulting feature map F1/2, has a size of H/2×W/2×128"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class BasicBlock(nn.Module):
    """ResNet BasicBlock（减半通道版本）"""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ModifiedResNet18(nn.Module):
    """
    改进版 ResNet-18：
    1. 去除初始 MaxPool，保持 H/2 分辨率
    2. 通道数减半：64→32, 128→64, 256→128, 512→256
    3. 初始卷积从 7×7 stride=2 变为 7×7 stride=2（保留，但去掉 MaxPool）

    输出各层特征用于 FPN：
    - C2 (layer1): H/2 × W/2 × 32  (stride=2)
    - C3 (layer2): H/4 × W/4 × 64  (stride=4)
    - C4 (layer3): H/8 × W/8 × 128 (stride=8)
    - C5 (layer4): H/16 × W/16 × 256 (stride=16)
    """

    def __init__(self):
        super().__init__()
        # 初始卷积层（7×7, stride=2, 无MaxPool）
        # 通道数减半：64 → 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # 注意：移除 MaxPool，仅 conv1 提供 stride=2

        # Layer1: 32 channels, stride=1 (保持 H/2 分辨率)
        self.layer1 = nn.Sequential(
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1)
        )  # 输出: H/2 × W/2 × 32

        # Layer2: 64 channels, stride=2
        self.layer2 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1)
        )  # 输出: H/4 × W/4 × 64

        # Layer3: 128 channels, stride=2
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )  # 输出: H/8 × W/8 × 128

        # Layer4: 256 channels, stride=2
        self.layer4 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )  # 输出: H/16 × W/16 × 256

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with C2, C3, C4, C5 feature maps
        """
        x = self.relu(self.bn1(self.conv1(x)))    # H/2 × W/2 × 32
        c2 = self.layer1(x)                         # H/2 × W/2 × 32
        c3 = self.layer2(c2)                         # H/4 × W/4 × 64
        c4 = self.layer3(c3)                         # H/8 × W/8 × 128
        c5 = self.layer4(c4)                         # H/16 × W/16 × 256
        return {'C2': c2, 'C3': c3, 'C4': c4, 'C5': c5}


class FPNSplit(nn.Module):
    """
    Feature Pyramid Network for Split Model

    输入：多尺度特征图 {C2, C3, C4, C5}
    输出：P2（H/2 × W/2 × 128）作为 F1/2

    FPN 自顶向下融合，输出通道均为 128
    """

    def __init__(self, in_channels_list: List[int] = [32, 64, 128, 256],
                 out_channels: int = 128):
        """
        Args:
            in_channels_list: 各层输入特征通道数 [C2, C3, C4, C5]
            out_channels: FPN 输出通道数（论文中为128）
        """
        super().__init__()
        self.out_channels = out_channels

        # 侧向连接：1×1 卷积统一通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # 输出卷积：3×3 平滑去除上采样伪影
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {'C2': ..., 'C3': ..., 'C4': ..., 'C5': ...}

        Returns:
            P2: H/2 × W/2 × 128 = F1/2
        """
        c2, c3, c4, c5 = features['C2'], features['C3'], features['C4'], features['C5']

        # 侧向连接
        p5 = self.lateral_convs[3](c5)  # H/16 × W/16 × 128
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        # 输出卷积（仅使用P2）
        p2 = self.output_convs[0](p2)

        return p2   # H/2 × W/2 × 128 = F1/2


class BackboneSplit(nn.Module):
    """
    Split Model 完整骨干网络 = ModifiedResNet18 + FPNSplit
    输出 F1/2: H/2 × W/2 × 128
    """

    def __init__(self, fpn_out_channels: int = 128):
        super().__init__()
        self.resnet = ModifiedResNet18()
        self.fpn = FPNSplit(
            in_channels_list=[32, 64, 128, 256],
            out_channels=fpn_out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, 3, H, W)，H=W=960

        Returns:
            F1/2: (B, 128, H/2, W/2) = (B, 128, 480, 480)
        """
        features = self.resnet(x)
        f_half = self.fpn(features)
        return f_half
