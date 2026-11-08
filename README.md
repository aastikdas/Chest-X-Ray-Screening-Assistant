
---

## 📌 Project Overview
This notebook is designed for Google Colab to train and evaluate a Chest X-ray classification model (DenseNet-121) with **Grad-CAM explainability** and **temperature scaling calibration**.

### Key Features:
1. **Zero Local Download**: Datasets are linked directly in Google Colab using `kagglehub`.
2. **Google Drive Checkpointing**: The trained model is automatically saved to `/content/drive/MyDrive/chest_xray_densenet.pth`. If the session disconnects or is refreshed, it automatically loads the saved model and **skips retraining**!
3. **Class Imbalance Handling**: Uses inverse-frequency class weights in the loss function.
4. **Model Calibration**: Measures and improves Expected Calibration Error (ECE) using Temperature Scaling.
5. **ROC & Confusion Matrix**: Computes multi-class AUROC and generates clear test evaluation charts.
6. **Horizontal Grad-CAM Visualization**: For sample test images, displays a clean 3-column horizontal comparison:

---

## 🚀 How to Run in Google Colab

1. Open [Google Drive](https://drive.google.com).
| Cell | Purpose |
| :--- | :--- |
| **Cell 1** | Markdown Title & Summary |
| **Cell 2** | Mount Google Drive & Environment Setup |
| **Cell 3** | Direct Kaggle Dataset Ingestion (`kagglehub`) |
| **Cell 4** | Medical Transforms & Class Imbalance Handling |
| **Cell 5** | DenseNet-121 Model Architecture Setup |
| **Cell 6** | **Train or Load from Drive** (Skips retraining if model exists!) |
| **Cell 8** | Confusion Matrix & Multi-Class ROC Curves |
| **Cell 9** | **Horizontal Grad-CAM Visualization** (`Original` $\rightarrow$ `Heatmap` $\rightarrow$ `Overlay`) |