# Brain MRI Tumor Detection (Attempt 2 - v2 / new_code_v3)

This folder contains the updated image processing pipeline for baseline atlas generation, tumor detection, and visualization.

## Files
- `calculate_baseline.cu`: CUDA program for computing voxel-wise mean and standard deviation atlas volumes across baseline scans.
- `detect_tumor.cu`: CUDA program using canonical bounding-box alignment, 14-hemisphere baseline atlas scoring, ventricular horn suppression, and contralateral-guided level-set segmentation.
- `preprocess_export.py`: Python script to load NIfTI scans (`.nii.gz`), apply skull-stripping, and export them into 18-byte header `.bin` volumes.
- `visualize_results.py`: Python visualizer to generate multi-planar (Axial, Coronal, Sagittal) `.png` slice views and cohort overview montages.
- `verify_py.py`: Python script for checking output volumes.

## Data Formats
- Input volumes: 3D MRI scans in NIfTI format (`.nii.gz`) or preprocessed `.bin` files with an 18-byte header (`dim_z, dim_y, dim_x, vz, vy, vx`).
- Output: Segmented tumor mask binary volumes (`.bin`) and multi-planar visualization images (`.png`).
