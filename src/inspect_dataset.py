
from pathlib import Path
import numpy as np

DATASET = Path(r"D:\PS07_CT2MRI\dataset")
SLICES = DATASET / "slices"

print("Dataset exists:", DATASET.exists())
print("Slices folder exists:", SLICES.exists())

files = list(SLICES.glob("*.npz"))

print("\nTotal NPZ files:", len(files))
print("\nFirst 10 files:")

for file in files[:10]:
    print(file.name)

if files:
    sample = np.load(files[0])

    print("\nInspecting:", files[0].name)

    for key in sample.files:
        arr = sample[key]
        print(
            f"{key}: shape={arr.shape}, "
            f"dtype={arr.dtype}, "
            f"min={arr.min()}, max={arr.max()}"
        )
