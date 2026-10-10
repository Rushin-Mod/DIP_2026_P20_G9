# Week 3 – Basic Filtering and Methodology

This folder contains the **Week 3 work** for the Brain MRI Tumor Segmentation and Quantitative Analysis project.

## Work Completed

- Developed a **baseline-subtraction approach** using multiple tumor-free MRI scans.
- Used five normal MRI scans to characterize normal anatomical and intensity variations.
- Explored voxel-wise difference calculation and threshold-based abnormality detection.
- Planned abnormal-region masking and classical image-processing post-processing.
- Defined quantitative analysis using:
  - Detected voxel count
  - Tumor volume
  - Bounding-box dimensions
  - Centroid/location
- Experimented with **GPU-accelerated image processing using CUDA**.
- Identified limitations of Python-based GPU processing, particularly VRAM usage, Python loops, and memory transfers.
- Planned transition of computationally intensive operations to **C/C++ and CUDA**, while retaining Python for visualization and verification.

## Current Workflow

```text
Normal MRI Scans
      ↓
Baseline Construction
      ↓
Voxel-wise Difference
      ↓
Normal Variation Analysis
      ↓
Thresholding
      ↓
Abnormal Region Mask
      ↓
Classical Post-processing
      ↓
Tumor Localization
      ↓
Quantitative Analysis