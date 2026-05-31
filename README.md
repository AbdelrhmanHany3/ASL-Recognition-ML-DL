# American Sign Language (ASL) Recognition Project 🤟🤖

An end-to-end Computer Vision and Deep Learning project designed to classify the American Sign Language (ASL) dataset into 28 classes (A-Z, Space, and Nothing). This project benchmarks traditional Machine Learning models against Custom and SOTA Deep Learning architectures.

## 📊 Dataset Overview
- **Source:** Kaggle (kapillondhe/american-sign-language)
- **Classes (28):** `['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Nothing', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']`
- **Image Preprocessing:** 
  - Grayscale conversion.
  - Image thresholding using Otsu's Binarization (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`) to isolate the hand gesture.
  - Bitwise masking and resizing images to `64x64` or `96x96` pixels.
  - Data Normalization (scaling pixel values to [0, 1]).

---

## 🛠️ Model Architectures & Benchmarking Results

In this project, I evaluated multiple paradigms to find the optimal balance between computational efficiency and accuracy:

### 1. Traditional Machine Learning (Flattened $64 \times 64$ Inputs)
*   **Support Vector Machine (SVM):** Configured with a linear kernel.
    *   **Test Accuracy:** `87.5%`
*   **K-Nearest Neighbors (KNN):** Evaluated with $K=3$.
    *   **Test Accuracy:** `91.96%`

### 2. Custom Convolutional Neural Network (CNN)
*   A deeply regularized sequential architecture utilizing **Batch Normalization**, **Max Pooling**, and progressive **Dropout** rates (0.25 to 0.40) to combat overfitting.
*   Implemented optimization strategies like `EarlyStopping` and `ReduceLROnPlateau`.
    *   **Test Accuracy:** `99.11%`
    *   **Test Loss:** `0.0242`

### 3. State-of-The-Art Transfer Learning
*   **VGG16 (Fine-Tuned):** Leveraged pre-trained ImageNet weights. The base model was frozen initially, followed by fine-tuning the top convolutional layers with a highly precise learning rate ($10^{-5}$).
    *   **Test Accuracy:** `100%` (Perfect convergence achieved with Augmentation)

---

## 📈 Performance Highlights (VGG16 & Custom CNN)
Both Deep Learning configurations exhibited outstanding metrics:
- **Precision / Recall / F1-Score:** Approaching `1.00` across almost all 28 gesture classes.
- **Robustness:** Enhanced via online image data augmentation (rotation, zoom, width/height shifts) while keeping horizontal flips disabled to protect sign orientation integrity.

---

## 🚀 How to Run

### Prerequisites
Install all core dependencies:
```bash
pip install kagglehub opencv-python scikit-learn tensorflow seaborn matplotlib 

### Dataset SetupThe notebooks are automated to pull directly from Kaggle, but you can also manually set up the environment variables:

Python
import os
os.environ['KAGGLE_USERNAME'] = "your_username"
os.environ['KAGGLE_KEY'] = "your_key"
### Execution
Run the Jupyter Notebooks sequentially:
1. `notebooks/AI_Practical_Project_Processing.ipynb` - For Preprocessing, SVM, and KNN.
2. `notebooks/ASL_ML_and_DL_Benchmarks.ipynb` - For Custom CNN, VGG16, MobileNet, and Fine-Tuning.

---
```
## 👨‍💻 Author
- **Abdelrhman Hany**
- GitHub: [@AbdelrhmanHany3](https://github.com/AbdelrhmanHany3)
