# Brain MRI Tumor Detection (Attempt 1)

This folder contains the image processing code for baseline atlas generation, tumor detection, and visualization.

## Files
- `calculate_baseline.cu`: CUDA program that computes mean and standard deviation volumes across baseline scans and outputs `.bin` volumes.
- `detect_tumor.cu`: CUDA program for tumor detection using statistical baseline subtraction, bilateral symmetry difference, and region growing.
- `preprocess_export.py`: Python script to load NIfTI scans (`.nii.gz`), apply skull-stripping, and export them into 18-byte header `.bin` volumes.
- `visualize_results.py`: Python visualizer to generate multi-planar (Axial, Coronal, Sagittal) `.png` slice views from the segmented `.bin` files.
- `verify_py.py`: Python script for checking output volumes.

## Data Formats
- Input volumes: 3D MRI scans in NIfTI format (`.nii.gz`) or preprocessed `.bin` files with an 18-byte header (`dim_z, dim_y, dim_x, vz, vy, vx`).
- Output: Segmented tumor mask binary volumes (`.bin`) and multi-planar visualization images (`.png`).
