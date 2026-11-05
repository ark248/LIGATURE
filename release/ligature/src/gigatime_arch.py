"""Re-implementation of the published GigaTIME network (nested U-Net, Valanarasu et al., Cell 2025).

Written from the authors' public description so that the released state dict loads
by parameter name; no downloaded code is executed. Weights are loaded with
torch.load(weights_only=True), i.e. tensors only.
"""
import torch
from torch import nn


class VGGBlock(nn.Module):
    def __init__(self, cin, cmid, cout):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(cin, cmid, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(cmid)
        self.conv2 = nn.Conv2d(cmid, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))


class NestedUNet(nn.Module):
    def __init__(self, num_classes=23, input_channels=3):
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = VGGBlock(input_channels, f[0], f[0])
        self.conv1_0 = VGGBlock(f[0], f[1], f[1])
        self.conv2_0 = VGGBlock(f[1], f[2], f[2])
        self.conv3_0 = VGGBlock(f[2], f[3], f[3])
        self.conv4_0 = VGGBlock(f[3], f[4], f[4])
        self.conv0_1 = VGGBlock(f[0] + f[1], f[0], f[0])
        self.conv1_1 = VGGBlock(f[1] + f[2], f[1], f[1])
        self.conv2_1 = VGGBlock(f[2] + f[3], f[2], f[2])
        self.conv3_1 = VGGBlock(f[3] + f[4], f[3], f[3])
        self.conv0_2 = VGGBlock(f[0] * 2 + f[1], f[0], f[0])
        self.conv1_2 = VGGBlock(f[1] * 2 + f[2], f[1], f[1])
        self.conv2_2 = VGGBlock(f[2] * 2 + f[3], f[2], f[2])
        self.conv0_3 = VGGBlock(f[0] * 3 + f[1], f[0], f[0])
        self.conv1_3 = VGGBlock(f[1] * 3 + f[2], f[1], f[1])
        self.conv0_4 = VGGBlock(f[0] * 4 + f[1], f[0], f[0])
        self.final = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x):
        u, p = self.up, self.pool
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(p(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, u(x1_0)], 1))
        x2_0 = self.conv2_0(p(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, u(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, u(x1_1)], 1))
        x3_0 = self.conv3_0(p(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, u(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, u(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, u(x1_2)], 1))
        x4_0 = self.conv4_0(p(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, u(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, u(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, u(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, u(x1_3)], 1))
        return self.final(x0_4)


def load_gigatime(path, num_classes=23):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    net = NestedUNet(num_classes)
    net.load_state_dict(state, strict=True)
    return net
