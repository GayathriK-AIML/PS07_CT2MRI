
import torch
from resunet import ResidualUNet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ResidualUNet(
    in_channels=3,
    out_channels=1,
    base_channels=16
).to(device)

# Test one batch
x = torch.randn(
    1, 3, 192, 224,
    device=device
)

model.eval()

with torch.no_grad():
    prediction = model(x)

print("Model:", model.__class__.__name__)
print("Device:", device)
print("Input shape:", x.shape)
print("Output shape:", prediction.shape)
print("Output range:",
      prediction.min().item(),
      prediction.max().item())

print("Parameters:",
      sum(p.numel() for p in model.parameters()))

print("GPU memory allocated:",
      round(torch.cuda.memory_allocated() / 1024**2, 2),
      "MB")
