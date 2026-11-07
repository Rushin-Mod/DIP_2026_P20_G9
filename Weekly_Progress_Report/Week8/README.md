# Week 8 – Removing Border False Detections and Improving Tumor Volume Estimation

**Project:** Brain MRI Tumor Segmentation and Quantitative Analysis Using Classical Image Processing  
**Course:** ECE501 - Digital Image Processing  
**Group:** G9 - The Medics

## Team
- Kush Kelaiya — AU2440018
- Rushin Modi — AU2440028
- Kritee Shah — AU2440080
- Jash Odedara — AU2440096

## Work Completed
- Investigated the root cause of the unwanted border garbage values seen in Week 7 and found that different patient scans include different amounts of the neck, shifting the brain up or down even when the 3D matrix dimensions match.
- Implemented bilateral brain-mask union during skull stripping so that dark tumors touching the outer edge of the brain are not accidentally cut off by morphological erosion.
- Added brain-bounding-box alignment to crop and scale only the isolated brain tissue of every patient scan into the exact same standard coordinate box before baseline comparison.
- Combined statistical Z-score baseline subtraction (using 7 healthy baseline brains and their left-right reflections) with bilateral mirror-difference scoring to better distinguish real tumors from normal brain folds and central fluid cavities (ventricles).
- Replaced fixed region-growing cutoffs with an adaptive region-growing threshold based on the intensity difference and spread between the tumor side and the opposite healthy hemisphere, followed by Gaussian surface smoothing and physical volume calculation in milliliters ($\text{mL}$).
- Achieved accurate tumor detection and volume estimation on a large portion of the dataset ($89.7\%$ hemisphere accuracy, $84.6\%$ lobe accuracy, and $70.7\%$ average volume accuracy across the 39 unique scans).
- Identified that there are still some false reports in tumor detection (such as midline or thalamic tumors pushing the central ventricles and causing the opposite wall to be detected) as well as partial volume mismatches on faint, low-contrast tumors that still need to be fixed and improved.

## Tools and Technologies
- Python
- C++
- CUDA
- Bilateral Brain-Mask Preservation
- Brain Bounding-Box Alignment and 3D Resampling
- Z-score Healthy Baseline Atlas Subtraction
- Bilateral Mirror-Asymmetry Scoring
- Contralateral-Guided Adaptive Region Growing
- Gaussian Surface Smoothing and Volumetric Analysis

## Next Week
- Focus on improving our approach to fix the remaining false detection reports (such as midline/ventricular shift cases) and partial volume mismatches so we can produce the best results.
- Complete the interactive GUI that we have started working on and integrate it with our processing system for a seamless user experience.
- Find another reference MRI dataset, prioritizing one that includes binary ground-truth masks on the segmented tumor areas so we can directly compare our outputs and derive the exact precision of our approach.
- Review and verify every aspect of the pipeline, documentation, and presentation since next week is the final week of our project.

## Files
- `Week8_G9.pdf`
