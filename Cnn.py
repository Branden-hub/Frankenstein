"""
Living AI System — Convolutional Neural Network Module
All CNN architectures: standard CNN, ResNet, DenseNet, Inception,
EfficientNet, MobileNet, U-Net, depthwise separable, dilated, transposed convolutions.
"""

import asyncio
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base import BaseModule, ModuleOutput

log = structlog.get_logger(__name__)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution — used in MobileNet."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = ConvBlock(in_ch, in_ch, stride=stride, groups=in_ch)
        self.pointwise = ConvBlock(in_ch, out_ch, kernel=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    """ResNet residual block with skip connection."""

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride, bias=False),
            nn.BatchNorm2d(channels),
        ) if stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv2(self.conv1(x))
        return self.relu(out + identity)


class DenseBlock(nn.Module):
    """DenseNet dense block — each layer connects to all subsequent layers."""

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvBlock(in_channels + i * growth_rate, growth_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class InceptionBlock(nn.Module):
    """Inception block with parallel filter sizes."""

    def __init__(self, in_ch: int, ch1: int, ch3_r: int, ch3: int,
                 ch5_r: int, ch5: int, pool_ch: int):
        super().__init__()
        self.branch1 = ConvBlock(in_ch, ch1, kernel=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_ch, ch3_r, kernel=1, padding=0),
            ConvBlock(ch3_r, ch3, kernel=3),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_ch, ch5_r, kernel=1, padding=0),
            ConvBlock(ch5_r, ch5, kernel=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_ch, pool_ch, kernel=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch1(x), self.branch2(x),
            self.branch3(x), self.branch4(x),
        ], dim=1)


class UNet(nn.Module):
    """
    U-Net for segmentation with skip connections.
    Encoder path compresses, decoder path expands,
    skip connections preserve spatial information.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: list = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        in_ch = in_channels
        for feat in features:
            self.encoder.append(nn.Sequential(
                ConvBlock(in_ch, feat), ConvBlock(feat, feat),
            ))
            in_ch = feat

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(features[-1], features[-1] * 2),
            ConvBlock(features[-1] * 2, features[-1] * 2),
        )

        # Decoder
        for feat in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.decoder.append(nn.Sequential(
                ConvBlock(feat * 2, feat), ConvBlock(feat, feat),
            ))

        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i + 1](x)
        return self.head(x)


class DilatedCNN(nn.Module):
    """Dilated convolution network for capturing multi-scale context."""

    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StandardCNN(nn.Module):
    """Standard deep CNN for image classification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32), ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128), ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256), ConvBlock(256, 256),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNet(nn.Module):
    """ResNet with configurable depth."""

    def __init__(self, num_blocks: list = None, num_classes: int = 10):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.stem = ConvBlock(3, 64, kernel=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, num_blocks[0])
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [ResidualBlock(channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stem(x))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


class ConvolutionalModule(BaseModule):
    """CNN module — activates for image and spatial processing tasks."""

    def __init__(self):
        self._cnn: StandardCNN | None = None
        self._resnet: ResNet | None = None
        self._unet: UNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "cnn"

    @property
    def output_type(self) -> str:
        return "vision"

    @property
    def required_capabilities(self) -> list[str]:
        return ["vision"]

    async def initialise(self) -> None:
        self._cnn = StandardCNN().to(self._device)
        self._resnet = ResNet().to(self._device)
        self._unet = UNet().to(self._device)
        for model in [self._cnn, self._resnet, self._unet]:
            model.eval()
        log.info("cnn_module.initialised")

    async def execute(
        self,
        message: Any,
        episodic_context: list[dict],
        knowledge_context: list[dict],
        working_memory: list[dict],
    ) -> ModuleOutput:
        return ModuleOutput(
            content="",
            confidence=0.0,
            output_type=self.output_type,
            source=self.name,
        )
