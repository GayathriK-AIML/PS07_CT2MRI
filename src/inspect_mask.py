
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(r"D:\PS07_CT2MRI")
VOLUMES = ROOT / "dataset" / "volumes"

PATIENTS = ["P1046", "P2658"]

for pid in PATIENTS:

    print(f"\n========== {pid} ==========")

    mask_path = VOLUMES / pid / "mask.nii.gz"

    mask = np.asarray(
        nib.load(str(mask_path)).dataobj
    )

    print("Shape:", mask.shape)
    print("Unique values:", np.unique(mask))

    occupancy = np.mean(mask > 0)

    print("Overall mask occupancy:", round(float(occupancy), 4))

    print("\nSlice-wise mask statistics:")

    counts = np.sum(mask > 0, axis=(0, 1))

    nonempty = np.where(counts > 0)[0]
    empty = np.where(counts == 0)[0]

    print("Total slices:", mask.shape[2])
    print("Non-empty slices:", len(nonempty))
    print("Empty slices:", len(empty))

    print("First non-empty slice:", nonempty[0] if len(nonempty) else None)
    print("Last non-empty slice:", nonempty[-1] if len(nonempty) else None)

    print("Minimum non-empty slice voxels:",
          int(counts[nonempty].min()) if len(nonempty) else 0)

    print("Maximum slice voxels:", int(counts.max()))

    print("\nSelected slice counts:")

    for z in [20, 40, 60, 80, 100, 120, 140, 160, 180]:
        if z < mask.shape[2]:
            print(f"Slice {z}: {int(counts[z])} mask voxels")
