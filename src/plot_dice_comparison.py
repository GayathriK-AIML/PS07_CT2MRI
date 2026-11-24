
import os
import matplotlib.pyplot as plt

PROJECT_ROOT = r"D:\PS07_CT2MRI"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Actual mean patient-level macro Dice scores
models = ["Residual U-Net", "Pix2Pix"]
dice_scores = [0.2451, 0.1117]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    dice_scores,
    width=0.5,
    edgecolor="black"
)

plt.title(
    "Anatomical Segmentation Dice Comparison",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Deep Learning Models", fontsize=12)
plt.ylabel("Mean Dice Score", fontsize=12)

plt.ylim(0, 0.30)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for bar, score in zip(bars, dice_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{score:.4f}",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

save_path = os.path.join(
    OUTPUT_DIR,
    "anatomical_dice_comparison.png"
)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("Dice comparison plot saved:", save_path)
