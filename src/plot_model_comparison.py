
import os
import matplotlib.pyplot as plt
import numpy as np

# Output directory
PROJECT_ROOT = r"D:\PS07_CT2MRI"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Actual validation results
models = ["Residual U-Net", "Pix2Pix"]

metrics = {
    "Masked MAE": [0.111834, 0.116208],
    "SSIM": [0.563518, 0.629526],
    "PSNR (dB)": [17.930539, 17.574280]
}

# Plot each evaluation metric separately
for metric_name, values in metrics.items():

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        models,
        values,
        width=0.5,
        edgecolor="black"
    )

    plt.title(
        f"Model Comparison - {metric_name}",
        fontsize=14,
        fontweight="bold"
    )

    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel("Deep Learning Models", fontsize=12)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Display exact values above bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11
        )

    plt.tight_layout()

    filename = metric_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(
        OUTPUT_DIR,
        f"model_comparison_{filename}.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved: {save_path}")

print("\nAll model comparison plots generated successfully!")
