
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch

from torch.amp import autocast
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(r"D:\PS07_CT2MRI")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dataset"))

from resunet import ResidualUNet
from multi_patient_loader import MultiPatientDataset
from normalise import normalise_ct, normalise_mri

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = ROOT / "models" / "pix2pix" / "best_generator.pth"

OUTPUT_DIR = ROOT / "outputs" / "pix2pix_best_epoch5"
RESULTS_DIR = ROOT / "results" / "pix2pix_best_epoch5"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_BATCH_SIZE = 8

# --------------------------------------------------
# 2. LOAD VALIDATION PATIENTS
# --------------------------------------------------

print("Device:", DEVICE)

val_dataset = MultiPatientDataset(
    ROOT / "val_patients.csv"
)

print("Validation patients:", len(val_dataset.patient_ids))

# --------------------------------------------------
# 3. LOAD BEST GENERATOR CHECKPOINT
# --------------------------------------------------

generator = ResidualUNet(
    in_channels=3,
    out_channels=1,
    base_channels=16
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

generator.load_state_dict(
    checkpoint["generator_state_dict"]
)

generator.eval()

print("Loaded checkpoint:", CHECKPOINT_PATH)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint masked MAE:", checkpoint["val_masked_mae"])

# --------------------------------------------------
# 4. FULL-VOLUME INFERENCE
# --------------------------------------------------

@torch.inference_mode()
def predict_patient(patient_id):

    folder = ROOT / "dataset" / "volumes" / patient_id

    ct_img = nib.load(str(folder / "ct.nii.gz"))
    mr_img = nib.load(str(folder / "mr.nii.gz"))
    mask_img = nib.load(str(folder / "mask.nii.gz"))

    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    mr = np.asarray(mr_img.dataobj, dtype=np.float32)
    mask = np.asarray(mask_img.dataobj, dtype=np.uint8)

    if not (ct.shape == mr.shape == mask.shape):
        raise ValueError(f"Shape mismatch: {patient_id}")

    # Use the same normalization as training
    ct = normalise_ct(ct)
    mr = normalise_mri(mr, mask)

    depth = ct.shape[2]

    prediction = np.zeros(ct.shape, dtype=np.float32)

    for start in range(0, depth, VAL_BATCH_SIZE):

        end = min(start + VAL_BATCH_SIZE, depth)

        batch_slices = []

        for z in range(start, end):

            previous_z = max(z - 1, 0)
            next_z = min(z + 1, depth - 1)

            ct_stack = np.stack([
                ct[:, :, previous_z],
                ct[:, :, z],
                ct[:, :, next_z]
            ], axis=0)

            batch_slices.append(ct_stack)

        batch = torch.from_numpy(
            np.stack(batch_slices).astype(np.float32)
        ).to(DEVICE)

        with autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):
            output = generator(batch)

        output = output.float().cpu().numpy()

        for i, z in enumerate(range(start, end)):
            prediction[:, :, z] = output[i, 0]

    prediction = np.clip(prediction, 0, 1)

    # --------------------------------------------------
    # 5. CALCULATE VALIDATION METRICS
    # --------------------------------------------------

    full_mae = float(
        np.mean(np.abs(prediction - mr))
    )

    mask_bool = mask > 0

    masked_mae = float(
        np.mean(np.abs(prediction[mask_bool] - mr[mask_bool]))
    ) if mask_bool.any() else np.nan

    ssim_values = []

    for z in range(depth):

        score = structural_similarity(
            mr[:, :, z],
            prediction[:, :, z],
            data_range=1.0
        )

        ssim_values.append(score)

    mean_ssim = float(np.mean(ssim_values))

    psnr = float(
        peak_signal_noise_ratio(
            mr,
            prediction,
            data_range=1.0
        )
    )

    # --------------------------------------------------
    # 6. SAVE PREDICTED MRI
    # --------------------------------------------------

    output_img = nib.Nifti1Image(
        prediction.astype(np.float32),
        ct_img.affine,
        ct_img.header.copy()
    )

    output_img.set_data_dtype(np.float32)

    output_path = OUTPUT_DIR / f"{patient_id}_predicted_mri.nii.gz"

    nib.save(output_img, str(output_path))

    print(f"\nPatient: {patient_id}")
    print(f"Full MAE: {full_mae:.6f}")
    print(f"Masked MAE: {masked_mae:.6f}")
    print(f"SSIM: {mean_ssim:.6f}")
    print(f"PSNR: {psnr:.4f}")
    print("Saved:", output_path)

    return {
        "patient_id": patient_id,
        "full_volume_mae": full_mae,
        "masked_mae": masked_mae,
        "mean_slice_ssim": mean_ssim,
        "full_volume_psnr": psnr
    }

# --------------------------------------------------
# 7. RUN ALL VALIDATION PATIENTS
# --------------------------------------------------

results = []

for patient_id in val_dataset.patient_ids:

    results.append(
        predict_patient(patient_id)
    )

# --------------------------------------------------
# 8. SAVE METRICS
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(
    RESULTS_DIR / "validation_metrics.csv",
    index=False
)

print("\n===================================")
print("EPOCH 5 INFERENCE COMPLETED")
print("===================================")

print(df.to_string(index=False))

print("\nAverage metrics:")
print(df.drop(columns=["patient_id"]).mean())

print("\nPredictions saved in:", OUTPUT_DIR)
print("Metrics saved in:", RESULTS_DIR)
