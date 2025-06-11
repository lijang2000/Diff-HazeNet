import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze_out = self.avg_pool(x).view(batch_size, channels)
        excitation = self.fc1(squeeze_out)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch_size, channels, 1, 1)
        return x * excitation


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Mid(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv_redim_dcp = nn.Conv2d(3, in_channels, kernel_size=1)
        self.conv_redim_bcp = nn.Conv2d(3, in_channels, kernel_size=1)
        self.Dp_wise = DepthwiseSeparableConv(in_channels=in_channels)
        self.se_block = SEBlock(in_channels=in_channels)
        self.Ap = nn.AdaptiveAvgPool2d(output_size=(8, 8))

    def forward(self, x, x_dcp, x_bcp):
        # x_shape = x.shape  # B, C, H, W

        x_dp = self.se_block(x)
        x_dp = self.Dp_wise(x_dp)

        x_dp = F.relu(x_dp)

        x_dcp = self.conv_redim_dcp(x_dcp)
        x_bcp = self.conv_redim_bcp(x_bcp)

        x_dcp = self.Ap(x_dcp)
        x_bcp = self.Ap(x_bcp)

        x_dcp_dp = x_dcp * x_dp
        x_bcp_dp = x_bcp * x_dp

        out = x_dcp_dp + x_bcp_dp + x_dp

        return out
