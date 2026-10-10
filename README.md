# DIP_2026_P20_G9 The Medics

ECE501 Digital Image Processing Project

## Problem Statement
Brain MRI Tumor Segmentation and Quantitative
Analysis Using Classical Image Processing.\
**Detailed Description**: This project analyzes brain MRI scans and analysis whether a tumor is present or not. If a tumor is present, we analyze the following parameters:
- Approximate location of the tumor
- Approximate size of the tumor
- The region in which the tumor is present (ex. frontal lobe, hypothalamus, etc.)

## Methodology and Approach
Our method of analyzing and detecting tumors is done in the following steps: 
- Loading & Slicing MRI Data: Opening 3D .nii.gz scans as standard arrays and slicing them into 2D cross-sections across different MRI contrast channels.
- Noise Cleanup: cleaning any noise that may interfere with the image detection.
- Contrast Boosting & Filtering: Using contrast adjustments and frequency filters to make faint tumor tissue pop out clearly from healthy brain matter.
- Tumor Outlining & Grouping: Finding sharp tumor edges with edge filters and grouping connected pixels together to isolate the mass.
- Size, Depth & Brain Region Calculation: Using pixel spacing and distance formulas to calculate the tumor's approximate volume, center point, and location within brain regions.

## Dataset
For our project, we will be using the publicly available dataset provided by OpenNeuro: [Retrospective Task/Rest fMRI Data from Tumor Patients
](https://openneuro.org/datasets/ds005003/versions/2.0.0)

## Team Members
- [**Jash Odedara**](https://github.com/JashOd30) (Team Leader)
- [**Rushin Modi**](https://github.com/Rushin-Mod)
- [**Kritee Shah**](https://github.com/kriteeshah723-ui)
- [**Kush Kelaiya**](https://github.com/Kush-Kelaiya22)