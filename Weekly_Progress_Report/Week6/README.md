# Week 6 – Alignment Recorrection, Dimension Matching, and Basic Filtering

**Project:** Brain MRI Tumor Segmentation and Quantitative Analysis Using Classical Image Processing  
**Course:** ECE501 - Digital Image Processing  
**Group:** G9 - The Medics

## Team
- Kush Kelaiya — AU2440018
- Rushin Modi — AU2440028
- Kritee Shah — AU2440080
- Jash Odedara — AU2440096

## Work Completed
- Worked on correcting alignment issues and matching the dimensions of MRI scans to enable accurate subtraction.
- Achieved successful subtraction after matching dimensions, but identified volume mismatches and limitations in detection accuracy.
- Observed that upscaling caused additional regions, including the skull, to be included in the processed output.
- Identified false positives around the skull and eye sockets, along with false negatives in which tumor regions were missed.
- Experimented with Laplacian filtering to improve tumor detection, but the results were not sufficiently accurate.
- Explored brain isolation and preprocessing techniques, including affine alignment, 3D resampling, and trilinear interpolation.
- Investigated morphological operations and distance transforms for skull stripping.
- Worked on centering and intensity scaling to improve consistency between MRI scans.
- Developed a subtraction pipeline involving:
  - Combinatorial baseline variance.
  - 3D Laplacian edge detection.
  - Two-level subtraction gating.
  - Left-right symmetry checking.

## Tools and Technologies
- Python
- C++
- CUDA
- 3D MRI Image Processing
- Affine Alignment and Resampling
- Trilinear Interpolation
- Laplacian Filtering
- Morphological Operations
- Distance Transform

## Next Week
- Experiment with Gaussian and arithmetic filtering techniques.
- Test different filter window sizes, including 3×3, 5×5, and 7×7.
- Improve detection accuracy by reducing false positives and false negatives.
- Refine the preprocessing and subtraction pipeline.
- Continue working on the GUI for the project.

## Files
- `Week6_G9.pdf`
