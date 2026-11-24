
import sys
import random
from pathlib import Path
from bisect import bisect_right

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, Sampler


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(r"D:\PS07_CT2MRI")
VOLUMES = ROOT / "dataset" / "volumes"

sys.path.insert(0, str(ROOT / "dataset"))

from normalise import normalise_ct, normalise_mri

BATCH_SIZE = 4
SEED = 42


# --------------------------------------------------
# 2. Multi-patient dataset
# --------------------------------------------------

class MultiPatientDataset(Dataset):

    def __init__(self, csv_path):

        self.patient_ids = (
            pd.read_csv(csv_path)["id"].astype(str).tolist()
        )

        self.patient_slices = []
        self.offsets = [0]

        self.cached_patient = None
        self.cached_data = None

        print("\nPreparing patient slice indexes...")

        for pid in self.patient_ids:

            mask_path = VOLUMES / pid / "mask.nii.gz"

            mask_img = nib.load(str(mask_path))
            mask = np.asarray(mask_img.dataobj)

            # Only use slices with valid neighbouring CT slices
            all_slices = list(range(1, mask.shape[2] - 1))

            # Slices containing mask voxels
            masked_slices = [
                z for z in all_slices
                if np.any(mask[:, :, z] > 0)
            ]

            # Completely empty-mask slices
            empty_slices = [
                z for z in all_slices
                if not np.any(mask[:, :, z] > 0)
            ]

            # Select every third empty-mask slice
            background_slices = empty_slices[::3]

            # Combine and sort all selected slices
            valid_slices = sorted(
                masked_slices + background_slices
            )

            self.patient_slices.append(valid_slices)

            self.offsets.append(
                self.offsets[-1] + len(valid_slices)
            )

            print(
                f"{pid}: "
                f"{len(masked_slices)} masked + "
                f"{len(background_slices)} background = "
                f"{len(valid_slices)} selected slices"
            )

        print("\nTotal patients:", len(self.patient_ids))
        print("Total samples:", len(self))

    def __len__(self):
        return self.offsets[-1]

    def load_patient(self, patient_index):

        pid = self.patient_ids[patient_index]

        # Reuse the current patient's volumes when possible
        if self.cached_patient == pid:
            return self.cached_data

        folder = VOLUMES / pid

        ct = np.asarray(
            nib.load(str(folder / "ct.nii.gz")).dataobj,
            dtype=np.float32
        )

        mr = np.asarray(
            nib.load(str(folder / "mr.nii.gz")).dataobj,
            dtype=np.float32
        )

        mask = np.asarray(
            nib.load(str(folder / "mask.nii.gz")).dataobj,
            dtype=np.float32
        )

        # Verify volume dimensions
        if not (ct.shape == mr.shape == mask.shape):
            raise ValueError(
                f"Volume shape mismatch for patient {pid}: "
                f"CT={ct.shape}, MRI={mr.shape}, Mask={mask.shape}"
            )

        # Official dataset normalization
        ct = normalise_ct(ct)
        mr = normalise_mri(mr, mask)

        self.cached_patient = pid
        self.cached_data = (ct, mr, mask)

        return self.cached_data

    def __getitem__(self, index):

        patient_index = bisect_right(
            self.offsets, index
        ) - 1

        local_index = index - self.offsets[patient_index]

        z = self.patient_slices[patient_index][local_index]

        ct, mr, mask = self.load_patient(patient_index)

        # Three neighbouring CT slices
        ct_input = np.stack([
            ct[:, :, z - 1],
            ct[:, :, z],
            ct[:, :, z + 1]
        ], axis=0)

        # Central MRI target
        mr_target = mr[:, :, z][None, :, :]

        # Corresponding mask
        mask_slice = mask[:, :, z][None, :, :]

        return (
            torch.from_numpy(ct_input.copy()),
            torch.from_numpy(mr_target.copy()),
            torch.from_numpy(mask_slice.copy())
        )


# --------------------------------------------------
# 3. Patient-grouped batch sampler
# --------------------------------------------------

class PatientBatchSampler(Sampler):

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle_patients=True
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_patients = shuffle_patients

    def __iter__(self):

        patient_order = list(
            range(len(self.dataset.patient_ids))
        )

        if self.shuffle_patients:
            random.shuffle(patient_order)

        for patient_index in patient_order:

            start = self.dataset.offsets[patient_index]
            end = self.dataset.offsets[patient_index + 1]

            indices = list(range(start, end))

            # Shuffle slices within this patient
            if self.shuffle_patients:
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):

        total_batches = 0

        for slices in self.dataset.patient_slices:

            total_batches += (
                len(slices) + self.batch_size - 1
            ) // self.batch_size

        return total_batches


# --------------------------------------------------
# 4. Create training and validation loaders
# --------------------------------------------------

if __name__ == "__main__":

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_dataset = MultiPatientDataset(
        ROOT / "train_patients.csv"
    )

    val_dataset = MultiPatientDataset(
        ROOT / "val_patients.csv"
    )

    train_sampler = PatientBatchSampler(
        train_dataset,
        BATCH_SIZE,
        shuffle_patients=True
    )

    val_sampler = PatientBatchSampler(
        val_dataset,
        BATCH_SIZE,
        shuffle_patients=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=0
    )

    # Test one training batch
    ct, mr, mask = next(iter(train_loader))

    print("\n========== TRAINING LOADER ==========")

    print("Training patients:", len(train_dataset.patient_ids))
    print("Training samples:", len(train_dataset))
    print("Training batches:", len(train_loader))

    print("\n========== VALIDATION LOADER ==========")

    print("Validation patients:", len(val_dataset.patient_ids))
    print("Validation samples:", len(val_dataset))
    print("Validation batches:", len(val_loader))

    print("\n========== SAMPLE BATCH ==========")

    print("CT:", ct.shape)
    print("MRI:", mr.shape)
    print("Mask:", mask.shape)

    print("CT range:", ct.min().item(), ct.max().item())
    print("MRI range:", mr.min().item(), mr.max().item())
    print("Mask unique values:", torch.unique(mask))

    print("\nGPU available:", torch.cuda.is_available())

    print("\nMulti-patient DataLoader test completed!")
