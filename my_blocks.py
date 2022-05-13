import math

from torch import nn
import torch
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1, bias = False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)


class norm_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class UNet__(nn.Module):
    def __init__(self, n_channels, n_classes, fix_channel = 4, n = 2, input_res = 16, output_res = 16, bilinear = True,
                 position_encoding = False):
        super(UNet__, self).__init__()
        self.input_res = input_res
        self.output_res = output_res
        self.n = n
        self.normup_n = int(math.log2(output_res / input_res))
        self.position_encoding = None
        if position_encoding is True:
            n_channels = n_channels + 32
            import PosEncoding.SinusoidalPositionalEncoding as fun_posEncoding
            self.register_buffer('position_encoding', tensor = fun_posEncoding(32, 256, 256))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, fix_channel)
        factor = 2 if bilinear else 1
        i = -1
        for i in range(n - 1):
            setattr(self, f'down{i + 1}', Down(fix_channel * (2 ** i), fix_channel * (2 ** (i + 1))))
        i = i + 1
        setattr(self, f'down{i + 1}', Down(fix_channel * (2 ** i), fix_channel * (2 ** (i + 1)) // factor))

        i = -1
        for i in range(n - 1):
            setattr(self, f'up{i + 1}',
                    Up(fix_channel * (2 ** (n - i)), fix_channel * (2 ** (n - i - 1)) // factor, bilinear))
        i = i + 1
        setattr(self, f'up{i + 1}', Up(fix_channel * 2, fix_channel, bilinear))

        for i in range(self.normup_n):
            setattr(self, f'norm_up{i + 1}', norm_Up(fix_channel, fix_channel, bilinear))

        self.outc1 = OutConv(fix_channel, n_classes - 1)

        self.outc2 = OutConv(fix_channel, 1)

    def forward(self, x):
        x = self.inc(x)
        downs = []
        for i in range(self.n):
            downs.append(x)
            x = getattr(self, f'down{i + 1}')(x)

        for i in range(self.n):
            x = getattr(self, f'up{i + 1}')(x, downs.pop())

        image = self.outc1(x)

        for i in range(self.normup_n):
            x = getattr(self, f'norm_up{i + 1}')(x)
        mask = torch.sigmoid(self.outc2(x))
        return image, mask
