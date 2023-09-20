import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6):
        super(Generator, self).__init__()

        # Downsample
        self.conv1 = nn.Conv2d(
            input_nc,
            64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
            padding_mode="reflect",
        )
        # self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        self.relu3 = nn.ReLU(inplace=False)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(n_residual_blocks)]
        )

        # Upsample
        self.conv4 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.in4 = nn.InstanceNorm2d(128, affine=True)
        self.relu4 = nn.ReLU(inplace=False)
        self.conv5 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.in5 = nn.InstanceNorm2d(64, affine=True)
        self.relu5 = nn.ReLU(inplace=False)
        self.conv6 = nn.Conv2d(
            64, output_nc, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Downsample
        x = self.conv1(x)
        # x = self.in1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.in3(x)
        x = self.relu3(x)

        # Residual blocks
        x = self.residual_blocks(x)

        # Upsample
        x = self.conv4(x)
        x = self.in4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.in5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.tanh(x)

        return x


class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, output_shape):
        super(PatchDiscriminator, self).__init__()

        # define architecture
        self.seq = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False),
        )

        # get the output tensor shape
        self.output_shape = self._get_conv_output_size(output_shape)

    def _get_conv_output_size(self, input_size):
        _input = torch.zeros(1, self.seq[0].in_channels, *input_size)
        output = self.forward(_input)
        return output.size()[1:]

    def forward(self, x):
        x = self.seq(x)
        return x


class ResBlockConcat(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockConcat, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv_skip = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn_skip = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        skip = self.conv_skip(x)
        skip = self.bn_skip(skip)

        out = torch.cat((out, skip), dim=1)
        out = F.relu(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.in2(out)

        # out += residual
        out = out + residual
        return out


class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()

        # 512@2x2 -> 32@2x2
        self.conv1 = nn.Conv2d(512, 32, (2, 2), 2, 1)
        # 512@2x2 cat 32@2x2 -> 32@2x2
        self.conv2 = nn.Conv2d(544, 32, (2, 2), 2, 1)
        # 512@2x2 cat 32@2x2 cat 32@2x2 -> 32@2x2
        self.conv3 = nn.Conv2d(576, 32, (2, 2), 2, 1)
        # 512@2x2 cat 32@2x2 cat 32@2x2 cat 32@2x2 -> 32@2x2
        self.conv4 = nn.Conv2d(608, 32, (2, 2), 2, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(torch.cat([x, c1], dim=1))
        c3 = self.conv3(torch.cat([x, c1, c2], dim=1))
        c4 = self.conv4(torch.cat([x, c1, c2, c3], dim=1))
        out = torch.cat([x, c1, c2, c3, c4], dim=1)

        return out


class DenseUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseUnet, self).__init__()

        # -----------DOWNSCALE-----------
        # [in_channels]@256x256 ->64@128x128
        self.d1 = nn.Conv2d(in_channels, 64, (5, 5), 2, 2)
        # 64@128x128 -> 128@64x64
        self.d2 = nn.Conv2d(64, 128, (5, 5), 2, 2)
        # 128@64x64 -> 256@32x32
        self.d3 = nn.Conv2d(128, 256, (5, 5), 2, 2)
        # 256@32x32 -> 512@16x16
        self.d4 = nn.Conv2d(256, 512, (5, 5), 2, 2)
        # 512@16x16 -> 512@8x8
        self.d5 = nn.Conv2d(512, 512, (5, 5), 2, 2)
        # 512@8x8 -> 512@4x4
        self.d6 = nn.Conv2d(512, 512, (5, 5), 2, 2)
        # 512@4x4 ->512@2x2
        self.d7 = nn.Conv2d(512, 512, (5, 5), 2, 2)

        # ----------DENSE BLOCK----------
        self.dense_block = DenseBlock()

        # ------------UPSCALE------------
        # 512@2x2 cat 640@2x2 -> 512@4x4
        self.u1 = nn.ConvTranspose2d(1152, 512, (6, 6), 2, padding=2)
        # 512@4x4 cat 512@4x4 -> 512@8x8
        self.u2 = nn.ConvTranspose2d(1024, 512, (6, 6), 2, padding=2)
        # 512@8x8 cat 512@8x8 -> 512@16x16
        self.u3 = nn.ConvTranspose2d(1024, 512, (6, 6), 2, padding=2)
        # 512@16x16 cat 512@16x16 -> 256@32x32
        self.u4 = nn.ConvTranspose2d(1024, 256, (6, 6), 2, padding=2)
        # 256@32x32 cat 256@32x32 -> 128@64x64
        self.u5 = nn.ConvTranspose2d(512, 128, (6, 6), 2, padding=2)
        # 128@64x64 cat 128@64x64 -> 64@128x128
        self.u6 = nn.ConvTranspose2d(256, 64, (6, 6), 2, padding=2)
        # 64@128x128 cat 64@128x128 -> [out_channels]@256x256
        self.u7 = nn.ConvTranspose2d(128, out_channels, (6, 6), 2, padding=2)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)

        dense_block = self.dense_block(d7)

        u1 = self.u1(torch.cat([dense_block, d7], dim=1))
        u2 = self.u2(torch.cat([u1, d6], dim=1))
        u3 = self.u3(torch.cat([u2, d5], dim=1))
        u4 = self.u4(torch.cat([u3, d4], dim=1))
        u5 = self.u5(torch.cat([u4, d3], dim=1))
        u6 = self.u6(torch.cat([u5, d2], dim=1))
        u7 = self.u7(torch.cat([u6, d1], dim=1))

        return u7


class PatchDiscriminatorPaired(nn.Module):
    def __init__(self):
        super(PatchDiscriminatorPaired, self).__init__()

        self.net = nn.Sequential(
            # 3@256x256 cat 1@256x256 -> ...
            nn.ConvTranspose2d(4, 4, (2, 2), 2),
            # ... -> 4@256x256
            nn.AvgPool2d((2, 2), 2),
            # 4@256x256 -> 64@128x128
            nn.Conv2d(4, 64, (5, 5), 2, 2),
            # 64@128x128 -> 128@64x64
            nn.Conv2d(64, 128, (5, 5), 2, 2),
            # 128@64x64 -> 256@32x32
            nn.Conv2d(128, 256, (5, 5), 2, 2),
            # 256@32x32 -> 512@16x16
            nn.Conv2d(256, 512, (5, 5), 2, 2),
            # 512@16x16 -> 512@8x8
            nn.Conv2d(512, 512, (5, 5), 2, 2),
            # 512@8x8 -> 1@8x8
            nn.Conv2d(512, 1, (5, 5), 1, 2),
        )

    def forward(self, x):
        nir, rgb = x
        out = self.net(torch.cat([nir, rgb], dim=1))

        return out
