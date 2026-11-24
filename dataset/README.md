---
license: other
pretty_name: "PS07 CT to MRI Conversion (student data)"
tags: [hackathon, ct2mri]
---
# PS07: CT to MRI Conversion

**Theme:** Model Training | 24-hour hackathon, teams of two, AI coding tools allowed.

## What you build
Generate an MRI-like brain volume from a CT scan, with a per-voxel reliability map, for a hospital the model has not seen.

## What is in this repository
120 brain CT/MRI pairs with body masks from two hospitals (centres A and B) on a common 192x224x192, 1 mm grid, as NIfTI volumes and as 2D slices, plus `normalise.py`.

## Output format
For each test patient `<id>/mri.nii.gz` (normalised 0 to 1) and `<id>/reliability.nii.gz` (higher = trust less), on the CT grid.

## Source and licence
SynthRAD2023 (Thummerer et al.), CC BY-NC 4.0. Cropped to the common grid by the organizers. Use this data only for the event and for non-commercial work, with attribution. Hidden test data and answer keys are not included.
Read the student problem statement document for the full task, scoring and rules.
