#  Chest X-Ray Screening Assistant with Explainability
> An explainable deep learning decision-support system for multi-class chest radiograph triage (Normal, Pneumonia, Tuberculosis) with Grad-CAM visual localization and confidence calibration.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chest-xray-assistant.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

---

## 🌐 Live Web Application
You can test the interactive web application live without installing anything:  
👉 **[chest-xray-assistant.streamlit.app](https://chest-xray-assistant.streamlit.app/)**

---

## 📌 Key Features
- **Tri-Class Triage:** Accurately classifies frontal chest radiographs (PA/AP views) into **Normal**, **Pneumonia**, or **Tuberculosis**.
- **High Diagnostic Accuracy:** Pretrained ResNet-50 achieves **97.46%** test accuracy (with 100% sensitivity on Tuberculosis).
- **Grad-CAM Lesion Localization:** Generates visual attention heatmaps overlaid on the original radiograph to explain where the model focused (e.g., focal consolidations, apical cavitary lesions).
- **Confidence Calibration & Uncertainty:** Uses post-hoc Platt temperature scaling ($T = 1.15$) and Shannon entropy to prevent overconfident misclassifications and flag ambiguous cases for secondary review.
- **Built-in Sample Scans:** Includes sample radiographs in `samples/` for immediate one-click testing directly from the sidebar.

---

## 📊 Model Performance

Evaluated on an independent test cohort ($N = 315$ balanced scans):

| Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| DenseNet-121 | 96.19% | 96.39% | 96.19% | 96.21% |
| **ResNet-50 (Selected)** | **97.46%** | **97.51%** | **97.46%** | **97.47%** |

---

## 🛠️ Local Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/aastikdas/Chest-X-Ray-Screening-Assistant.git
cd Chest-X-Ray-Screening-Assistant
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 📓 Model Training (Google Colab)

To retrain the models or reproduce the experimental benchmarks:
1. Open the included notebook: `Chest_XRay_Screening.ipynb`.
2. Upload it to [Google Colab](https://colab.research.google.com).
3. Set the hardware accelerator to **T4 GPU** (`Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`).
4. Click **Runtime** $\rightarrow$ **Run all**. The notebook automatically downloads the curated public datasets via `kagglehub`, trains both architectures, and computes calibration and Grad-CAM metrics.

---

## 📁 Repository Structure

```text
├── app.py                                # Streamlit web application
├── requirements.txt                      # Project dependencies
├── chest_xray_resnet50_2100_e8_clean.pth  # Trained model checkpoint (ResNet-50)
├── Chest_XRay_Screening.ipynb            # Google Colab training & evaluation notebook
├── samples/                              # Built-in sample radiographs for demo
├── .streamlit/
│   └── config.toml                       # Streamlit server and theme configurations
└── README.md                             # Project documentation
```

---

## ⚠️ Clinical Disclaimer
This software is an academic research prototype designed for clinical decision-support demonstration. It is not an FDA/CE cleared diagnostic device. All AI assessments should be validated by certified medical professionals.