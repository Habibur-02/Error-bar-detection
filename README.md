# Error-bar-detection
solution of home engineering task on error bar detection in scientific plots using matplotlib and ML,DL algorithms
# 📉 Robust Error Bar Detection in Scientific Plots

A complete pipeline for detecting error bars in scanned, noisy, and low-resolution scientific plots. This project demonstrates a journey from **Traditional Computer Vision (DIP)** to an optimized **Deep Learning (CNN)** approach, achieving a **74.32% accuracy** on real-world data.

---

## 📊 Performance Comparison (The "Why AI?" Proof)

I started with traditional image processing concepts but found them insufficient for noisy data. Deep Learning provided the required robustness.

| Approach | Method Used | Accuracy (<5px) | MAE (Error) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Traditional DIP (Canny Edge) | 43.89% | 19.02 px | ❌ Failed on noise |
| **Final Model** | **CNN + Augmentation** | **74.32%** | **5.03 px** | ✅ **Success** |

---

## 🛠️ The Engineering Journey (Step-by-Step Approach)

My development process followed an iterative engineering cycle to solve the "Domain Gap" problem:

### 1. 🔍 Initial Attempt: Traditional DIP (Baseline)
Before using heavy models, I attempted a lightweight solution using **Canny Edge Detection** and pixel scanning.
* **Result:** It worked on clean images but failed significantly on scanned plots with text overlaps and broken lines.
* **Decision:** Shifted to a learning-based approach (CNN).

### 2. 💾 Data Generation
Real-world data was scarce (only 150 images).
* I wrote a script (`dataset_generator.py`) to generate **3,000 synthetic plots**.
* *Note:* The synthetic dataset is hosted externally due to size. [Link to Google Drive Dataset](#) *(Optional)*.

### 3. 📉 Phase 1: Training on Synthetic Data
I trained a Patch-based CNN model (`best_model.pth`) on the synthetic data.
* **Challenge:** When tested on real scanned images, performance dropped drastically due to the **Domain Gap** (Synthetic data was "too clean").

### 4. 📈 Phase 2: Transfer Learning (Fine-Tuning)
To bridge the gap, I applied **Transfer Learning**.
* I fine-tuned the model using the 150 real images (`fine_tune.py`).
* **Result:** Accuracy increased significantly as the model learned to handle noise.

### 5. 🚀 Phase 3: Advanced Optimization (Augmentation)
To push for the highest precision, I implemented **Data Augmentation** (Random Rotation +/- 5°, Brightness Scaling).
* **Result:** This produced the final model (`final_model_advanced.pth`) with **74.32% accuracy** and robust performance on rotated/scanned documents.

---

## 📂 Project Structure

```text
├── dataset_generator.py       # Script to create 3000 synthetic images
├── train.py                   # Phase 1: Base training script
├── fine_tune.py               # Phase 2: Transfer learning script
├── fine_tune_advance.py       # Phase 3: Advanced training with Augmentation
├── inference.py               # Main script to generate predictions
├── evalaute.py                # Script to calculate Accuracy & MAE
├── visualize.py               # Tool to draw detections on images
├── best_model.pth             # Model trained on Synthetic Data
├── final_model.pth            # Model after Fine-Tuning
├── final_model_advanced.pth   # Best Model (With Augmentation)
└── dip/                       # Folder for Traditional CV (Canny Edge) code
    ├── dip_baseline.py
    └── evaluate_dip.py
```
## 💻 How to Run (Step-by-Step Guide)

Follow this sequence to reproduce the engineering journey from the Baseline (DIP) to the Final Deep Learning Model.

### ✅ Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
```
1️⃣ Step 1: The Baseline Experiment (Traditional DIP)

Before using heavy Deep Learning models, I first tested if traditional Computer Vision (Canny Edge Detection) works on the provided 150 real images.

```
cd dip
# Run the heuristic algorithm
python dip_baseline.py

# Check the metrics
python evaluate_dip.py

```
Observation: You will see an accuracy of around 43%. The method fails on noisy and scanned images. (Go back to the main directory)
```
cd ..
```
2️⃣ Step 2: Data Preparation for Deep Learning

Since the 150 real images are not enough for training a CNN, I generated synthetic data.

``` # Generates 3,000 synthetic plots in 'dataset/' folder
python dataset_generator.py
```
3️⃣ Step 3: Training Pipeline (Iterative Improvement)

I trained the model in three phases to handle the domain gap.

Phase 1: Pre-training on Synthetic Data Trains the base model on the 3,000 generated images.
```
python train.py
```

Output: Saves best_model.pth.

Phase 2: Fine-Tuning on Real Data (Transfer Learning) Adapts the synthetic model to the 150 real scanned images.
```
python fine_tune.py
```
Output: Saves final_model.pth.

Phase 3: Advanced Optimization (Augmentation) Re-trains the model with Random Rotation (+/- 5°) and Brightness Scaling to make it robust against scanning artifacts.

```
python fine_tune_advance.py
```
Output: Saves final_model_advanced.pth (This is the Best Model).

4️⃣ Step 4: Final Inference & Evaluation

Now, use the best model (final_model_advanced.pth) to predict error bars on the real dataset.

Generate Predictions:
```
python inference.py

```
Output: JSON files will be generated in the output_predictions/ folder.

Calculate Metrics:

```
python evalaute.py
```
Expected Result:

    Accuracy (<5px): ~74.32% 🚀

    MAE: ~5.03 pixels

5️⃣ Step 5: Visualization (Visual Proof)

Draw the predicted error bars on the images to verify the quality visually.
```
python visualize.py
```
Output: Check the visualization_results/ folder. You will see red lines accurately drawn over the error bars, even in noisy images.
| Feature | Traditional DIP (Canny Edge) | Deep Learning (My CNN) |
| :--- | :--- | :--- |
| **Approach** | Pixel Scanning Heuristics | Patch-based Regression |
| **Handling Noise** | ❌ Fails on text/grid lines | ✅ Ignores noise/text |
| **Handling Scans** | ❌ Fails on broken lines | ✅ Robust to rotation/blur |
| **Accuracy** | 43.89% | **74.32%** |
| **Avg Error** | 19.02 px | **5.03 px** |


👨‍💻 Author

Md Habibur Rahman (Aasif) 
Rajshahi University of Engineering and Technology (RUET)
