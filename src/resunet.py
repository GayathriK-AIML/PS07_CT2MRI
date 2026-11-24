
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity

        return self.activation(out)


class ResidualUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1,
                 base_channels=16):

        super().__init__()

        c = base_channels

        # Encoder
        self.enc1 = ResidualBlock(in_channels, c)
        self.enc2 = ResidualBlock(c, c * 2)
        self.enc3 = ResidualBlock(c * 2, c * 4)
        self.enc4 = ResidualBlock(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(c * 8, c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            c * 16, c * 8, kernel_size=2, stride=2
        )
        self.dec4 = ResidualBlock(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(
            c * 8, c * 4, kernel_size=2, stride=2
        )
        self.dec3 = ResidualBlock(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(
            c * 4, c * 2, kernel_size=2, stride=2
        )
        self.dec2 = ResidualBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(
            c * 2, c, kernel_size=2, stride=2
        )
        self.dec1 = ResidualBlock(c * 2, c)

        # MRI intensity prediction
        self.output = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output constrained to [0, 1]
        return torch.sigmoid(self.output(d1))
