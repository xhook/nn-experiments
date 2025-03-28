from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        skip_after_nonlin: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_after_nonlin = skip_after_nonlin

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        if self.skip_after_nonlin:
            return self.relu(out) + identity
        else:
            return self.relu(out + identity)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        return out

class ResNetLayer(nn.Module):
    def __init__(
        self,
        inplanes: int,
        layer_configs: list[dict[str, Any]],
    ) -> None:
        super().__init__()

        self.layer_configs = layer_configs
        identity_adapters: list[nn.Module] = []
        layers = []
        for config in self.layer_configs:
            block: Type[Union[Bottleneck]] = config["block"]
            norm_layer: Callable[..., nn.Module] = config["norm_layer"]
            planes: int = config["planes"]
            blocks: int = config["blocks"]
            stride: int = config.get("stride", 1)
            groups: int = config.get("groups", 1)
            base_width: int = config.get("base_width", 64)
            
            self.blocks = blocks
            
            if stride != 1 or inplanes != planes * block.expansion:
                identity_adapters.append(nn.Sequential(
                    conv1x1(inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ))
            else:
                identity_adapters.append(nn.Identity())

            layers.append(
                block(
                    inplanes, planes, stride, groups, base_width, norm_layer
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, self.blocks):
                identity_adapters.append(nn.Identity())
                layers.append(
                    block(
                        inplanes,
                        planes,
                        groups=groups,
                        base_width=base_width,
                        norm_layer=norm_layer,
                    )
                )
        self.identity_adapters = nn.ModuleList(identity_adapters)
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.blocks):
            identity = self.identity_adapters[i](x)
            x = identity + self.layers[i](x)
        return x

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = ResNetLayer(self.inplanes, [
            {"block": block, "norm_layer": norm_layer, "planes": 64, "blocks": layers[0], "groups": groups, "base_width": width_per_group},
            {"block": block, "norm_layer": norm_layer, "planes": 128, "blocks": layers[1], "stride": 2, "groups": groups, "base_width": width_per_group},
            {"block": block, "norm_layer": norm_layer, "planes": 256, "blocks": layers[2], "stride": 2, "groups": groups, "base_width": width_per_group},
            {"block": block, "norm_layer": norm_layer, "planes": 512, "blocks": layers[3], "stride": 2, "groups": groups, "base_width": width_per_group},
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def resnet18(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs: Any) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

def resnext50_32x4d(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnext101_32x8d(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnext101_64x4d(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def wide_resnet50_2(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def wide_resnet101_2(**kwargs: Any) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)