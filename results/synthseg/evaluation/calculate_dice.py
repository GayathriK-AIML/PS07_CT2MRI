
import os
import csv
import numpy as np
import nibabel as nib

BASE = "/mnt/d/PS07_CT2MRI"
PATIENTS = ["P2658", "P6123", "P7363", "P7480", "P8947"]

DIRS = {
    "GroundTruth": f"{BASE}/results/synthseg/ground_truth",
    "ResidualUNet": f"{BASE}/results/synthseg/model1_resunet",
    "Pix2Pix": f"{BASE}/results/synthseg/model2_pix2pix"
}

OUT = f"{BASE}/results/synthseg/evaluation"
os.makedirs(OUT, exist_ok=True)

patient_results = []
structure_results = []

for patient in PATIENTS:
    print(f"\nEvaluating {patient}")

    paths = {
        model: f"{folder}/{patient}_seg.nii.gz"
        for model, folder in DIRS.items()
    }

    images = {name: nib.load(path) for name, path in paths.items()}
    arrays = {
        name: np.asarray(img.dataobj)
        for name, img in images.items()
    }

    gt = arrays["GroundTruth"]

    for name in ["ResidualUNet", "Pix2Pix"]:
        if arrays[name].shape != gt.shape:
            raise ValueError(f"{patient}: {name} shape mismatch")

        if not np.allclose(
            images[name].affine,
            images["GroundTruth"].affine,
            atol=1e-5
        ):
            raise ValueError(f"{patient}: {name} affine mismatch")

    # Include labels present in either segmentation.
    labels = np.union1d(
        np.unique(gt),
        np.unique(arrays["ResidualUNet"])
    )
    labels = np.union1d(labels, np.unique(arrays["Pix2Pix"]))
    labels = labels[labels != 0]

    result = {"Patient": patient}

    for model in ["ResidualUNet", "Pix2Pix"]:
        pred = arrays[model]
        dice_values = []

        total_intersection = 0
        total_denominator = 0

        for label in labels:
            gt_mask = gt == label
            pred_mask = pred == label

            intersection = np.count_nonzero(gt_mask & pred_mask)
            denominator = (
                np.count_nonzero(gt_mask)
                + np.count_nonzero(pred_mask)
            )

            dice = (
                2 * intersection / denominator
                if denominator > 0 else np.nan
            )

            dice_values.append(dice)
            total_intersection += intersection
            total_denominator += denominator

            structure_results.append({
                "Patient": patient,
                "Model": model,
                "Label": int(label),
                "GroundTruthVoxels": int(np.count_nonzero(gt_mask)),
                "PredictedVoxels": int(np.count_nonzero(pred_mask)),
                "Dice": dice
            })

        result[f"{model}_MacroDice"] = float(np.mean(dice_values))
        result[f"{model}_MicroDice"] = (
            2 * total_intersection / total_denominator
        )

        result[f"{model}_ForegroundFraction"] = float(
            np.count_nonzero(pred) / pred.size
        )

    result["GroundTruthForegroundFraction"] = float(
        np.count_nonzero(gt) / gt.size
    )

    patient_results.append(result)

    print("Residual U-Net macro Dice:",
          round(result["ResidualUNet_MacroDice"], 4))
    print("Pix2Pix macro Dice:",
          round(result["Pix2Pix_MacroDice"], 4))

def save_csv(filename, rows):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

save_csv(f"{OUT}/corrected_patient_dice.csv", patient_results)
save_csv(f"{OUT}/corrected_structure_dice.csv", structure_results)

print("\n========== CORRECTED DICE RESULTS ==========")

for model in ["ResidualUNet", "Pix2Pix"]:
    macro = np.mean([
        r[f"{model}_MacroDice"] for r in patient_results
    ])
    micro = np.mean([
        r[f"{model}_MicroDice"] for r in patient_results
    ])

    print(f"\n{model}")
    print(f"Mean patient macro Dice: {macro:.4f}")
    print(f"Mean patient micro Dice: {micro:.4f}")

print("\nSaved corrected evaluation files in:", OUT)
