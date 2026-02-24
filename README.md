# Retina Vessel Segmentation: SegFormer vs U-Net (MONAI)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-green)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-yellow)
![Segmentation](https://img.shields.io/badge/Segmentation-orange)
![Medical Imaging](https://img.shields.io/badge/Medical_Imaging-red)


**Comparative study of SegFormer and U-Net for blood vessel segmentation on retina images using CLAHE preprocessing.**

![Retina Example](https://user-images.githubusercontent.com/placeholder/retina_example.png)  
*Example of retina image preprocessing and vessel segmentation.*

---

## Description

This project focuses on **automatic segmentation of blood vessels in retina images**. The goal is to **compare the performance** of two deep learning architectures:  

- **U-Net (MONAI)** – optimized for medical imaging. 
- **SegFormer** – transformer-based architecture for semantic segmentation.  

**Preprocessing:**  
- Images are converted to **grayscale or green channel**.  
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is applied to enhance vessel visibility.  

**Dataset:**  
- Compatible with public retina datasets like **DRIVE**, **STARE**, and **CHASE_DB1**.

**Metrics:**
- Dice coefficient, IoU (MONAI metrics optimized for medical imaging).

---

## Features

- CLAHE preprocessing for better vessel contrast  
- U-Net (MONAI) & SegFormer comparison  
- Visualization of segmentation masks  
- Evaluation with medical imaging metrics (Dice, IoU)  
- Reproducible pipeline with notebooks & scripts  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/Retina-Vessel-Segmentation-SegFormer-U-Net.git
cd Retina-Vessel-Segmentation-SegFormer-U-Net
```

2. Install dependencies:
```python
pip install -r requirements.txt
```
---
## Usage

---

## Result 
---

## Results

---
| Model     | Dice Coefficient | IoU  |
| --------- | ---------------- | ---- |
| U-Net     | 0.81             | 0.73 |
| SegFormer | 0.85             | 0.78 |

## Project Structure
retina-vessel-segmentation/
│
├─ README.md
├─ requirements.txt
├─ data/              # dataset or links
├─ notebooks/         # exploration and visualization
├─ src/               # preprocessing, models, utils
├─ outputs/           # segmented images, results
└─ .gitignore

## Licence
License

This project is licensed under the MIT License.