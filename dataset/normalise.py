"""Fixed intensity normalisation for PS-07; the scorer uses these same functions."""
import argparse
import nibabel as nib
import numpy as np

CT_LO, CT_HI = -1000.0, 2000.0
MRI_PERCENTILE = 99.5


def normalise_ct(ct_hu):
    """CT in HU -> [0, 1]: clip to [-1000, 2000] HU, then scale linearly."""
    return ((np.clip(ct_hu, CT_LO, CT_HI) - CT_LO) / (CT_HI - CT_LO)).astype(np.float32)


def normalise_mri(mr, mask):
    """MRI -> [0, 1]: divide by the 99.5th percentile of MRI values inside the mask, clip to [0, 1]."""
    scale = np.percentile(mr[mask > 0], MRI_PERCENTILE)
    return np.clip(mr / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Normalise a CT or MRI NIfTI file")
    ap.add_argument("kind", choices=["ct", "mri"])
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--mask", help="mask NIfTI, required for mri")
    a = ap.parse_args()
    img = nib.load(a.inp)
    x = np.asarray(img.dataobj).astype(np.float32)
    if a.kind == "ct":
        y = normalise_ct(x)
    else:
        y = normalise_mri(x, np.asarray(nib.load(a.mask).dataobj))
    nib.save(nib.Nifti1Image(y, img.affine), a.out)
