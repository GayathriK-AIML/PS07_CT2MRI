
import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):

    def __init__(self, in_channels=4, base_channels=64):

        super().__init__()

        def block(in_ch, out_ch, stride, use_norm=True):

            layers = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=not use_norm
                )
            ]

            if use_norm:
                layers.append(nn.BatchNorm2d(out_ch))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return nn.Sequential(*layers)

        # Input: concatenated CT and MRI
        self.model = nn.Sequential(

            block(
                in_channels,
                base_channels,
                stride=2,
                use_norm=False
            ),

            block(
                base_channels,
                base_channels * 2,
                stride=2
            ),

            block(
                base_channels * 2,
                base_channels * 4,
                stride=2
            ),

            block(
                base_channels * 4,
                base_channels * 8,
                stride=1
            ),

            nn.Conv2d(
                base_channels * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, ct, mr):

        # Concatenate CT input and MRI candidate
        combined = torch.cat([ct, mr], dim=1)

        # Return raw logits (no sigmoid)
        return self.model(combined)


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    discriminator = PatchGANDiscriminator().to(device)

    # Test batch
    ct = torch.randn(1, 3, 192, 224).to(device)
    mr = torch.randn(1, 1, 192, 224).to(device)

    output = discriminator(ct, mr)

    print("Device:", device)
    print("CT input shape:", ct.shape)
    print("MRI input shape:", mr.shape)
    print("Discriminator output shape:", output.shape)
    print("Parameters:",
          sum(p.numel() for p in discriminator.parameters()))

    assert output.ndim == 4
    assert output.shape[0] == 1
    assert output.shape[1] == 1

    print("\nPatchGAN discriminator test PASSED!")
