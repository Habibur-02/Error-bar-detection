# Error-bar-detection
solution of home engineering task on error bar detection in scientific plots using matplotlib and ML,DL algorithms
# 📊 Error Bar Detection Pipeline

An automated computer vision pipeline designed to detect and predict the coordinates of upper and lower error bars in scientific plot images. This project utilizes a **Deep Learning approach (ResNet-18)** with a **Synthetic-to-Real Transfer Learning** strategy to achieve high precision even with limited real-world data.

---

## 🚀 Key Results
| Metric | Performance |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | **5.56 pixels** |
| **Accuracy (Error ≤ 5px)** | **80.76%** |
| **Accuracy (Error ≤ 10px)** | **88.50%** |
| **Architecture** | Fine-Tuned ResNet-18 (Regression Head) |

---

## 🧠 Methodology & Approach

To achieve state-of-the-art results with a limited provided dataset (~150 images), I implemented a **Two-Stage Training Pipeline**:

### 1. Data-Driven Synthetic Generation
Instead of relying solely on scarce real data, I developed a statistical generator (`dataset_generator.py`) that:
* Analyzes the real company dataset to understand property distributions (Cap sizes, line widths, noise levels).
* Generates **3,000 realistic synthetic images** mimicking these statistics.
* Simulates real-world imperfections: Gaussian blur, salt-and-pepper noise, overlapping lines, and text labels (ymin/ymax).

### 2. Deep Learning Architecture (ResNet-18)
* **Backbone:** ResNet-18 (Pre-trained architecture modified for Grayscale input).
* **Head:** Modified Fully Connected layers for **Regression** (predicting `top_dist` and `bottom_dist`).
* **Loss Function:** `SmoothL1Loss` (Huber Loss) – robust against outliers compared to MSE.
* **Optimization:** `AdamW` optimizer with a `ReduceLROnPlateau` scheduler for adaptive learning rates.

### 3. Transfer Learning (The "Winning" Strategy)
* **Phase 1 (Pre-training):** Trained on 3,000 synthetic images to learn general feature extraction (Edges, Caps, Bars).
* **Phase 2 (Fine-Tuning):** Transfer Learning applied on the 150 real company images with heavy augmentation to bridge the domain gap. This boosted accuracy from ~46% to **80.76%**.

---

## 📂 Project Structure

```text
Error-Bar-Detection/
├── dataset_generator.py       # Generates 3k synthetic images based on real stats
├── model_utils_resnet.py      # ResNet-18 Architecture & Dataset Loader
├── train_resnet.py            # Phase 1: Pre-training on Synthetic Data
├── fine_tune_resnet.py        # Phase 2: Transfer Learning on Real Data
├── inference_resnet.py        # Generates final JSON coordinates (Assignment 2)
├── evaluate.py                # Calculates MAE and Accuracy metrics
├── visualize.py               # Draws predicted error bars on images
├── requirements.txt           # Dependencies
├── error_bar_resnet_best.pth  # Pre-trained Model weights
├── final_interview_model.pth  # Final Fine-Tuned Model weights
│
├── dataset/                   # Generated Synthetic Data
├── company_dataset/           # Original Provided Data (Images + Labels)
├── assignment2_output_final/  # Final JSON Predictions
└── visualization_result/      # Annotated Output Images
