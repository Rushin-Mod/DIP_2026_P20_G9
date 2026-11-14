
# Week 9 - Completion and Improvements

## Brain MRI Tumor Segmentation and Quantitative Analysis Using Classical Image Processing

**Team:** G9 - The Medics

### Team Members
- Kush Kelaiya
- Rushin Modi
- Kritee Shah
- Jash Odedara

---

## 1. Work Done This Week

During Week 9, we focused on completing the base detection and segmentation of brain MRI scans and improving the accuracy and reliability of our system using classical image processing techniques.

### 1.1 Improving Tumor Detection
- Performed multiple iterations by modifying and experimenting with different formulas to improve tumor detection.
- Identified issues with accuracy calculation formulas that were producing false values.
- Adopted subtraction with absolute values to identify tumor regions.
- Compared detected regions with the `.json` values provided in the dataset.

### 1.2 Masking and Filtering Techniques
Explored different masking and filtering approaches to prevent unwanted regions from interfering with tumor detection.

Techniques explored:
- Skull Stripping
- Parenchymal Band Thresholding
- Central Intracranial Connected Components

These techniques helped distinguish relevant brain regions from other structures in the head region.

### 1.3 Accuracy Calculation

We experimented with different accuracy calculation approaches and selected:

**Bounded Volumetric Ratio + Hemisphere-Gated Spatial Decay**

This method was partially successful in detecting tumor regions across most scans in the dataset.

However, the base scan continued to produce negative accuracy values despite successful tumor detection, indicating inconsistencies in the current accuracy calculation method.

### 1.4 Current Performance

The approximate performance of our system is:

| Evaluation Metric | Accuracy |
|---|---:|
| Overall accuracy across images | 41.1% |
| Accuracy on scans containing tumors | 67.4% |

These values represent the current performance of our implementation.

---

## 2. Challenges Faced
- Achieving industry-standard accuracy using only classical image processing techniques.
- Identifying reliable accuracy calculation formulas.
- Preventing skeletal regions and surrounding brain matter from interfering with tumor segmentation.
- Correcting negative accuracy values observed in the base scan.
- Improving consistency across different MRI scans.

---

## 3. Results

The system was tested across multiple brain MRI scans to evaluate tumor detection and segmentation.

The results included both successful detections and false results, helping us identify areas requiring further improvement.

---

## 4. Future Improvements
- Investigate and correct inconsistencies in accuracy calculations.
- Improve tumor detection accuracy across the dataset.
- Refine masking and filtering techniques.
- Improve segmentation reliability while maintaining the constraint of using classical image processing methods only.

---

## 5. Technologies and Techniques Used

- Classical Image Processing
- Brain MRI Analysis
- Image Segmentation
- Skull Stripping
- Parenchymal Band Thresholding
- Connected Components
- Image Masking and Filtering
- Quantitative Analysis
- Python

---

## 6. Repository

**GitHub Repository:**  
[Brain MRI Tumor Segmentation - G9](https://github.com/Rushin-Mod/DIP)

---

**Project:** Brain MRI Tumor Segmentation and Quantitative Analysis Using Classical Image Processing  
**Week:** 9  
**Team:** G9 - The Medics
