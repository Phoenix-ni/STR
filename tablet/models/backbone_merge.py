"""
Merge Model 骨干网络：标准 ResNet-18 + FPN

对应论文 Section 3.2：
    "ResNet-18 combined with FPN where the number of channels in FPN is 256.
     The RoIAlign output size is 7×7"

    标准 ResNet-18（保留 MaxPool），FPN 输出 256 通道
    输出 P2: H/4 × W/4 × 256（最高分辨率FPN层）
    对应图中 "H/4×W/4×256"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class BasicBlock(nn.Module):
    """标准 ResNet BasicBlock"""
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
        return self.relu(out + identity)


class StandardResNet18(nn.Module):
    """
    标准 ResNet-18（保留 MaxPool，标准通道数）

    输出各层特征用于 FPN：
    - C2 (layer1): H/4 × W/4 × 64   (stride=4，包含MaxPool)
    - C3 (layer2): H/8 × W/8 × 128  (stride=8)
    - C4 (layer3): H/16 × W/16 × 256 (stride=16)
    - C5 (layer4): H/32 × W/32 × 512 (stride=32)
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        # 初始卷积 + MaxPool（标准ResNet）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # stride=4 total

        # Layer1: 64 channels, stride=1→total stride=4
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )

        # Layer2: 128 channels, stride=2→total stride=8
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )

        # Layer3: 256 channels, stride=2→total stride=16
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )

        # Layer4: 512 channels, stride=2→total stride=32
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)
        )

        if pretrained:
            self._load_pretrained()
        else:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained(self):
        """尝试加载 torchvision 预训练权重"""
        try:
            import torchvision.models as tv_models
            pretrained = tv_models.resnet18(weights='IMAGENET1K_V1')
            state_dict = pretrained.state_dict()
            self.load_state_dict(state_dict, strict=False)
            print("成功加载 ResNet-18 预训练权重")
        except Exception as e:
            print(f"无法加载预训练权重: {e}，使用随机初始化")
            self._init_weights()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))   # H/2 × W/2 × 64
        x = self.maxpool(x)                        # H/4 × W/4 × 64
        c2 = self.layer1(x)                        # H/4 × W/4 × 64
        c3 = self.layer2(c2)                        # H/8 × W/8 × 128
        c4 = self.layer3(c3)                        # H/16 × W/16 × 256
        c5 = self.layer4(c4)                        # H/32 × W/32 × 512
        return {'C2': c2, 'C3': c3, 'C4': c4, 'C5': c5}


class FPNMerge(nn.Module):
    """
    Feature Pyramid Network for Merge Model

    输入：多尺度特征图 {C2, C3, C4, C5}
    输出：P2（H/4 × W/4 × 256）用于 RoIAlign

    论文: "backbone for image feature extraction is a standard ResNet-18 + FPN,
           where the number of channels in FPN is 256"
    """

    def __init__(self, in_channels_list: List[int] = [64, 128, 256, 512],
                 out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # 侧向连接：1×1 卷积统一通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # 输出卷积：3×3 平滑
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
            P2: H/4 × W/4 × 256
        """
        c2, c3, c4, c5 = features['C2'], features['C3'], features['C4'], features['C5']

        # 自顶向下路径
        p5 = self.lateral_convs[3](c5)   # H/32 × W/32 × 256
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        # 输出卷积（仅使用P2）
        p2 = self.output_convs[0](p2)

        return p2   # H/4 × W/4 × 256


class BackboneMerge(nn.Module):
    """
    Merge Model 完整骨干网络 = StandardResNet18 + FPNMerge
    输出 P2: H/4 × W/4 × 256
    """

    def __init__(self, fpn_out_channels: int = 256, pretrained: bool = False):
        super().__init__()
        self.resnet = StandardResNet18(pretrained=pretrained)
        self.fpn = FPNMerge(
            in_channels_list=[64, 128, 256, 512],
            out_channels=fpn_out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, 3, H, W)，H=W=960

        Returns:
            P2: (B, 256, H/4, W/4) = (B, 256, 240, 240)
        """
        features = self.resnet(x)
        p2 = self.fpn(features)
        return p2
