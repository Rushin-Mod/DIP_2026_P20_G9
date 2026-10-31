# Week 7 – Further Improvement on False Negatives and Correction

**Project:** Brain MRI Tumor Segmentation and Quantitative Analysis Using Classical Image Processing  
**Course:** ECE501 - Digital Image Processing  
**Group:** G9 - The Medics

## Team
- Kush Kelaiya — AU2440018
- Rushin Modi — AU2440028
- Kritee Shah — AU2440080
- Jash Odedara — AU2440096

## Work Completed
- Shifted from raw left-right subtraction to Z-score-based intensity subtraction using a healthy brain baseline atlas.
- Implemented Difference of Gaussians (DoG) to help distinguish 3D tumor-like blobs from thin anatomical wrinkles.
- Explored a deficit operator and bilateral asymmetry to generate a neoplastic saliency score.
- Worked on targeted region growing using white matter barriers to improve tumor-region identification.
- Applied surface smoothing to refine the detected regions.
- Obtained partial results and continued refining the detection process.
- Identified remaining false detections and unwanted border values that require further correction.

## Tools and Technologies
- Python
- C++
- CUDA
- Z-score-Based Intensity Subtraction
- Healthy Brain Baseline Atlas
- Difference of Gaussians (DoG)
- Bilateral Asymmetry Analysis
- Region Growing
- Surface Smoothing

## Next Week
- Improve detection precision and reduce false positives and false negatives.
- Find a suitable MRI dataset containing ground-truth segmentation masks for validation.
- Remove incorrectly detected regions and refine segmentation boundaries.
- Improve GUI presets and functionality.
- Identify and fix remaining bugs in the processing pipeline.

## Files
- `Week7_G9.pdf`
