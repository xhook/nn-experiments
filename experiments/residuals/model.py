import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, depth: int, kernel_size: int, residual: bool):
        super().__init__()
        self.depth = depth
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=10, out_channels=10, kernel_size=kernel_size, padding='same') for i in range(depth)])
        self.residual = residual

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 0, 0, 0, 0, 7, 0, 0), mode='constant', value=0.0)
        for conv in self.convs:
            z = F.relu(conv(x))
            x = x + z if self.residual else z                
        x = torch.mean(x, (2, 3)) # global average pooling
        return x

class GenResNet(nn.Module):
    def __init__(self, depth: int, kernel_size: int):
        super().__init__()
        self.depth = depth
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=10, out_channels=10, kernel_size=kernel_size, padding='same') for i in range(depth)])
        self.layer_cons = nn.Parameter(torch.tril(torch.ones(depth + 1, depth + 1)) * (torch.rand((depth + 1, depth + 1)) + 0.5))

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 0, 0, 0, 0, 7, 0, 0), mode='constant', value=0.0)
        layer_outs = [x]
        for i in range(self.depth + 1):
            layer_x = sum([o * w for o, w in zip(layer_outs, self.layer_cons[i])])
            if i < self.depth:
                layer_outs.append(F.relu(self.convs[i](layer_x)))
            else:
                break
        x = torch.mean(layer_x, (2, 3)) # global average pooling
        return x
