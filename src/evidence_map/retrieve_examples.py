
from pathlib import Path

import numpy as np
import nibabel as nib


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset" / "volumes"

INDEX_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "evidence_map"
    / "patch_index_train.npz"
)

PATCH_SIZE = 32
TOP_K = 3
BATCH_SIZE = 10000


# ============================================================
# CT NORMALIZATION
# Same normalization as the official dataset normalise.py
# ============================================================

def normalize_ct(ct):
    ct = np.asarray(ct, dtype=np.float32)

    ct = np.clip(ct, -1000.0, 2000.0)

    return ((ct + 1000.0) / 3000.0).astype(np.float32)


# ============================================================
# EXTRACT QUERY PATCH FROM A VALIDATION PATIENT
# ============================================================

def extract_query_patch(patient_id):

    patient_dir = DATASET_DIR / patient_id

    ct_path = patient_dir / "ct.nii.gz"
    mask_path = patient_dir / "mask.nii.gz"

    if not ct_path.exists():
        raise FileNotFoundError(ct_path)

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    ct_img = nib.load(str(ct_path))
    mask_img = nib.load(str(mask_path))

    ct = ct_img.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata(dtype=np.float32)

    if ct.shape != mask.shape:
        raise ValueError("CT and mask shapes do not match.")

    if not np.allclose(ct_img.affine, mask_img.affine):
        raise ValueError("CT and mask affines do not match.")

    if ct.ndim != 3:
        raise ValueError("Expected a 3D CT volume.")

    ct = normalize_ct(ct)
    mask = mask > 0

    # Select the axial slice containing the largest mask area.
    slice_areas = np.sum(mask, axis=(0, 1))
    z = int(np.argmax(slice_areas))

    if slice_areas[z] == 0:
        raise ValueError("No valid mask found in this patient.")

    slice_mask = mask[:, :, z]

    # Find the centre of the mask on the selected slice.
    coordinates = np.argwhere(slice_mask)

    center_x = int(np.round(coordinates[:, 0].mean()))
    center_y = int(np.round(coordinates[:, 1].mean()))

    # Align patch coordinates to the training index grid.
    x0 = (center_x // PATCH_SIZE) * PATCH_SIZE
    y0 = (center_y // PATCH_SIZE) * PATCH_SIZE

    X, Y, _ = ct.shape

    # Ensure the patch remains inside the volume.
    x0 = min(x0, X - PATCH_SIZE)
    y0 = min(y0, Y - PATCH_SIZE)

    x0 = max(0, x0)
    y0 = max(0, y0)

    query_patch = ct[
        x0:x0 + PATCH_SIZE,
        y0:y0 + PATCH_SIZE,
        z
    ]

    query_mask = slice_mask[
        x0:x0 + PATCH_SIZE,
        y0:y0 + PATCH_SIZE
    ]

    if query_patch.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError("Invalid query patch dimensions.")

    body_fraction = float(np.mean(query_mask))

    if body_fraction < 0.5:
        raise ValueError(
            f"Query patch body fraction is too low: "
            f"{body_fraction:.3f}"
        )

    return query_patch, {
        "patient_id": patient_id,
        "slice_z": z,
        "patch_x": x0,
        "patch_y": y0,
        "body_fraction": body_fraction
    }


# ============================================================
# RETRIEVE SIMILAR CT PATCHES AND PAIRED MRI PATCHES
# ============================================================

def retrieve_evidence(
    query_ct_patch,
    index_file=INDEX_FILE,
    top_k=TOP_K,
    batch_size=BATCH_SIZE,
    exclude_patient_id=None
):

    query_ct_patch = np.asarray(
        query_ct_patch,
        dtype=np.float32
    )

    if query_ct_patch.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f"Expected a 32x32 query patch, "
            f"received {query_ct_patch.shape}"
        )

    if top_k < 1 or batch_size < 1:
        raise ValueError("top_k and batch_size must be positive.")

    index_file = Path(index_file)

    if not index_file.exists():
        raise FileNotFoundError(
            f"Training patch index not found: {index_file}"
        )

    # Normalize the query for normalized cross-correlation.
    query_vector = query_ct_patch.reshape(-1).copy()

    query_vector -= np.mean(query_vector)

    query_norm = np.linalg.norm(query_vector)

    if query_norm < 1e-8:
        raise ValueError("Query patch has almost no variation.")

    query_vector /= query_norm

    # Load the training index.
    with np.load(index_file, allow_pickle=False) as index:

        train_ct = index["ct_patches"]
        train_mr = index["mr_patches"]
        metadata = index["metadata"]

        if not (
            len(train_ct) == len(train_mr) == len(metadata)
        ):
            raise ValueError("Index arrays have different lengths.")

        number_of_patches = len(train_ct)

        if number_of_patches == 0:
            raise ValueError("Training patch index is empty.")

        # Calculate normalized cross-correlation in batches.
        all_scores = np.empty(
            number_of_patches,
            dtype=np.float32
        )

        for start in range(0, number_of_patches, batch_size):

            end = min(start + batch_size, number_of_patches)

            batch = train_ct[start:end].reshape(
                end - start, -1
            ).astype(np.float32)

            batch -= np.mean(
                batch,
                axis=1,
                keepdims=True
            )

            norms = np.linalg.norm(
                batch,
                axis=1,
                keepdims=True
            )

            norms[norms < 1e-8] = 1e-8

            batch /= norms

            all_scores[start:end] = batch @ query_vector

        # Exclude the query patient from the training matches.
        if exclude_patient_id is not None:

            patient_ids = metadata[:, 0].astype(str)

            same_patient = (
                patient_ids == str(exclude_patient_id)
            )

            all_scores[same_patient] = -np.inf

        valid_indices = np.flatnonzero(
            np.isfinite(all_scores)
        )

        if len(valid_indices) == 0:
            raise ValueError("No eligible training patches found.")

        selected_count = min(top_k, len(valid_indices))

        candidate_scores = all_scores[valid_indices]

        selected_positions = np.argpartition(
            candidate_scores,
            -selected_count
        )[-selected_count:]

        top_indices = valid_indices[selected_positions]

        top_indices = top_indices[
            np.argsort(all_scores[top_indices])[::-1]
        ]

        top_scores = all_scores[top_indices].copy()

        retrieved_ct = train_ct[top_indices].copy()
        retrieved_mr = train_mr[top_indices].copy()
        retrieved_metadata = metadata[top_indices].copy()

    # IMPORTANT:
    # Our index builder stores metadata as:
    # patient_id, slice_z, patch_x, patch_y

    matches = []

    for rank, (idx, score) in enumerate(
        zip(top_indices, top_scores),
        start=1
    ):

        patient_id, slice_z, patch_x, patch_y = (
            retrieved_metadata[rank - 1]
        )

        matches.append({
            "rank": rank,
            "patient_id": str(patient_id),
            "similarity": float(score),
            "z": int(slice_z),
            "x": int(patch_x),
            "y": int(patch_y),
            "index": int(idx)
        })

    return {
        "ct_patches": retrieved_ct,
        "mr_patches": retrieved_mr,
        "scores": top_scores,
        "metadata": retrieved_metadata,
        "matches": matches
    }


# ============================================================
# COMPLETE RETRIEVAL WORKFLOW
# ============================================================

def retrieve_for_patient(patient_id, top_k=TOP_K):

    query_patch, query_info = extract_query_patch(patient_id)

    evidence = retrieve_evidence(
        query_ct_patch=query_patch,
        top_k=top_k,
        exclude_patient_id=patient_id
    )

    return {
        "query_patch": query_patch,
        "query_info": query_info,
        "evidence": evidence
    }


# ============================================================
# TEST EXECUTION
# ============================================================

if __name__ == "__main__":

    patient_id = "P8947"

    print("=" * 55)
    print("EVIDENCE MAP RETRIEVAL TEST")
    print("=" * 55)

    result = retrieve_for_patient(patient_id, top_k=3)

    print("\nQUERY PATIENT INFORMATION")

    for key, value in result["query_info"].items():
        print(f"{key}: {value}")

    print("\nTOP RETRIEVED TRAINING REFERENCES")

    for match in result["evidence"]["matches"]:

        print(
            f"Rank {match['rank']} | "
            f"Patient: {match['patient_id']} | "
            f"Similarity: {match['similarity']:.4f} | "
            f"Slice: {match['z']} | "
            f"X: {match['x']} | "
            f"Y: {match['y']}"
        )

    print("\nRetrieved CT shape:",
          result["evidence"]["ct_patches"].shape)

    print("Retrieved MRI shape:",
          result["evidence"]["mr_patches"].shape)

    print("\nRETRIEVAL TEST COMPLETED")
