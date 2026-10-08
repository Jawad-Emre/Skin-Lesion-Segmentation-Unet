# Skin Lesion Segmentation: Traditional vs. Deep Learning

## 📌 Project Overview

This project presents a comparative study of **Traditional Digital Image Processing (DIP)** versus **Deep Learning (DL)** for the segmentation of skin lesions from dermoscopic images. Using the **ISIC 2018 Task 1 Dataset**, we evaluate the performance gap between unsupervised thresholding methods and supervised Convolutional Neural Networks (CNNs).

* **Goal:** To automate the precise segmentation of melanoma and other skin lesions.
* **Dataset:** ISIC 2018 Challenge (Task 1: Lesion Boundary Segmentation).
* **Key Finding:** Deep Learning (U-Net) outperformed Traditional Methods (Otsu) by a margin of **~10.5%** in Dice Coefficient.

---

## 📊 Comparative Results (The "Battle")

We evaluated both methods on a held-out test set of 415 images.

| Methodology | Technique | Dice Coefficient (Score) | Accuracy | Key Observation |
| --- | --- | --- | --- | --- |
| **Traditional (Baseline)** | Otsu's Thresholding | **0.7358** | ~70% | Struggles with low contrast, hair, and surgical markers. |
| **Deep Learning (Proposed)** | U-Net Architecture | **0.8129** | **92.85%** | Robust to artifacts; captures complex lesion shapes accurately. |

> **Conclusion:** While Otsu's method provides a fast, unsupervised baseline, the U-Net model demonstrates superior capability in handling the high variability of skin lesions, making it more suitable for clinical-grade analysis.

---

## 🖼️ Visual Results

### 1. Best vs. Worst Cases (U-Net)

* **Best Cases (Dice > 0.98):** The model perfectly segments dark, high-contrast lesions.
* **Worst Cases (Dice < 0.10):** The model struggles with images containing:
* Extreme hair occlusion (dense hair covering the lesion).
* Surgical ink markers (purple/black ink).
* Extremely low contrast (faint pink lesions).



### 2. Live Demo (Local Test)

* **Input:** A random internet image of a mole (with hair).
* **Output:** A clean, smooth binary mask.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Deep Learning:** TensorFlow / Keras
* **Image Processing:** OpenCV (`cv2`), NumPy, Matplotlib
* **Hardware:** Trained on NVIDIA Tesla P100 (Kaggle).

---

## 🚀 How to Run

### 1. Training (Kaggle/Colab)

Open the `skin-lesion-segmentation-unet.ipynb` notebook. Ensure you have the ISIC 2018 dataset added to your input path.

* **Input Path:** `/kaggle/input/isic2018-challenge-task1-data-segmentation`
* **Runtime:** GPU (Tesla T4 or P100 recommended).

### 2. Local Demo (Inference)

To run the pre-trained model on your own images:

1. **Install Dependencies:**
```bash
pip install tensorflow opencv-python numpy matplotlib

```


2. **Download Model:** Place `unet_skin_lesion_model.keras` in the project folder.
3. **Run Script:**
```bash
python project.ipynb

```


4. **Input:** When prompted, enter the filename of your test image (e.g., `test_mole.jpg`).

---

## 🔮 Future Work

To address the "Worst Case" failures identified in this study:

1. **DullRazor Algorithm:** Implement digital hair removal preprocessing before feeding images to the U-Net.
2. **Data Augmentation:** Increase training data with synthetic hair and marker artifacts to make the model robust to these specific noises.

---

## 📝 Author

* **Name:** Jawad Emre
* **Context:** Semester Project