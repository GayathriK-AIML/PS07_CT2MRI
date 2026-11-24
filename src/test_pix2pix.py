
import torch
import torch.nn as nn

from resunet import ResidualUNet
from patchgan_discriminator import PatchGANDiscriminator

# --------------------------------------------------
# 1. DEVICE
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# --------------------------------------------------
# 2. INITIALIZE BOTH NETWORKS
# --------------------------------------------------

generator = ResidualUNet(
    in_channels=3,
    out_channels=1,
    base_channels=16
).to(DEVICE)

discriminator = PatchGANDiscriminator(
    in_channels=4,
    base_channels=64
).to(DEVICE)

print("\nGenerator parameters:",
      sum(p.numel() for p in generator.parameters()))

print("Discriminator parameters:",
      sum(p.numel() for p in discriminator.parameters()))

# --------------------------------------------------
# 3. LOSS FUNCTIONS
# --------------------------------------------------

adversarial_loss = nn.BCEWithLogitsLoss()
reconstruction_loss = nn.L1Loss()

LAMBDA_L1 = 100

# --------------------------------------------------
# 4. OPTIMIZERS
# --------------------------------------------------

optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=2e-4,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=2e-4,
    betas=(0.5, 0.999)
)

# --------------------------------------------------
# 5. CREATE SAMPLE INPUT
# --------------------------------------------------

ct = torch.rand(
    1, 3, 192, 224,
    device=DEVICE
)

real_mri = torch.rand(
    1, 1, 192, 224,
    device=DEVICE
)

# --------------------------------------------------
# 6. GENERATOR FORWARD PASS
# --------------------------------------------------

generator.train()
discriminator.train()

fake_mri = generator(ct)

print("\nGenerator output:", fake_mri.shape)
print("Generated MRI range:",
      fake_mri.min().item(),
      fake_mri.max().item())

assert fake_mri.shape == real_mri.shape

# --------------------------------------------------
# 7. DISCRIMINATOR TRAINING TEST
# --------------------------------------------------

optimizer_D.zero_grad(set_to_none=True)

real_prediction = discriminator(
    ct, real_mri
)

fake_prediction = discriminator(
    ct, fake_mri.detach()
)

real_labels = torch.ones_like(real_prediction)
fake_labels = torch.zeros_like(fake_prediction)

loss_D_real = adversarial_loss(
    real_prediction, real_labels
)

loss_D_fake = adversarial_loss(
    fake_prediction, fake_labels
)

loss_D = 0.5 * (loss_D_real + loss_D_fake)

loss_D.backward()
optimizer_D.step()

print("\nDiscriminator loss:", loss_D.item())

# --------------------------------------------------
# 8. GENERATOR TRAINING TEST
# --------------------------------------------------

optimizer_G.zero_grad(set_to_none=True)

# Prevent discriminator weights from updating
for parameter in discriminator.parameters():
    parameter.requires_grad_(False)

fake_prediction = discriminator(ct, fake_mri)

target_labels = torch.ones_like(fake_prediction)

loss_G_GAN = adversarial_loss(
    fake_prediction, target_labels
)

loss_G_L1 = reconstruction_loss(
    fake_mri, real_mri
)

loss_G = loss_G_GAN + LAMBDA_L1 * loss_G_L1

loss_G.backward()
optimizer_G.step()

# Restore discriminator gradients
for parameter in discriminator.parameters():
    parameter.requires_grad_(True)

print("Generator adversarial loss:", loss_G_GAN.item())
print("Generator L1 loss:", loss_G_L1.item())
print("Total generator loss:", loss_G.item())

# --------------------------------------------------
# 9. FINAL CHECKS
# --------------------------------------------------

assert torch.isfinite(loss_D)
assert torch.isfinite(loss_G)

assert fake_mri.min().item() >= 0
assert fake_mri.max().item() <= 1

print("\n================================")
print("PIX2PIX INTEGRATION TEST PASSED")
print("================================")
