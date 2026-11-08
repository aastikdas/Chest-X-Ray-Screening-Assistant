import json
import os

def create_simplified_notebook():
    cells = []

    def add_markdown(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # =========================================================================
    # CELL 1: Markdown Title & Project Goal
    # =========================================================================
    add_markdown("""# 🫁 Chest X-Ray Screening Assistant with Explainability
### Multi-Class Classification (Normal, Pneumonia, Tuberculosis) with Grad-CAM & Calibration

**Project Objective:**
- Flag likely **Pneumonia** or **Tuberculosis** in chest X-rays to support (not replace) clinicians.
- Show **WHERE** the model looked using **Grad-CAM heatmaps** overlaid on the original X-ray.
- Provide **calibrated confidence** and uncertainty metrics.
- Automatically save and load the trained model from **Google Drive** so training is not repeated if the session disconnects.""")

    # =========================================================================
    # CELL 2: Setup, Mount Google Drive & Imports
    # =========================================================================
    add_code("""# ==============================================================================
# Cell 1: Mount Google Drive & Environment Setup
# ==============================================================================
import os

# 1. Mount Google Drive so model weights are saved permanently
try:
    from google.colab import drive
    drive.mount('/content/drive')
    MODEL_SAVE_PATH = '/content/drive/MyDrive/chest_xray_densenet.pth'
    print("✅ Google Drive mounted successfully.")
except Exception:
    MODEL_SAVE_PATH = 'chest_xray_densenet.pth'
    print("ℹ️ Running in local/offline environment without Colab Drive mount.")

print(f"📁 Trained model checkpoint path: {MODEL_SAVE_PATH}")

# 2. Install dependencies (kagglehub for direct dataset linking, timm, opencv)
!pip install -q kagglehub opencv-python seaborn scikit-learn

import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Execution device: {device}")
""")

    # =========================================================================
    # CELL 3: Direct Kaggle Dataset Ingestion
    # =========================================================================
    add_code("""# ==============================================================================
# Cell 2: Direct Kaggle Dataset Ingestion (Cloud-to-Cloud Linking)
# ==============================================================================
import kagglehub

CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

print("🔄 Linking Kaggle datasets directly (no local file download needed)...")

# Public Kaggle datasets:
# 1. Tuberculosis (TB) Chest X-ray Database (Tawsifur Rahman et al.)
# 2. Chest X-ray Images (Pneumonia) (Paul Mooney et al.)
tb_path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
pneu_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("   TB dataset linked at:       ", tb_path)
print("   Pneumonia dataset linked at:", pneu_path)

records = []

# Collect Tuberculosis images
tb_files = glob.glob(os.path.join(tb_path, "**", "*Tuberculosis*", "*.png"), recursive=True) + \
           glob.glob(os.path.join(tb_path, "**", "*TB*", "*.png"), recursive=True) + \
           glob.glob(os.path.join(tb_path, "**", "*Tuberculosis*", "*.jpg"), recursive=True)
tb_files = list(set([f for f in tb_files if os.path.isfile(f)]))[:500]
for f in tb_files:
    records.append({"filepath": f, "label": "Tuberculosis", "target": 2})

# Collect Pneumonia images
pneu_files = glob.glob(os.path.join(pneu_path, "**", "PNEUMONIA", "*.jpeg"), recursive=True) + \
             glob.glob(os.path.join(pneu_path, "**", "PNEUMONIA", "*.png"), recursive=True)
pneu_files = list(set([f for f in pneu_files if os.path.isfile(f)]))[:500]
for f in pneu_files:
    records.append({"filepath": f, "label": "Pneumonia", "target": 1})

# Collect Normal images
norm_files = glob.glob(os.path.join(pneu_path, "**", "NORMAL", "*.jpeg"), recursive=True) + \
             glob.glob(os.path.join(tb_path, "**", "Normal", "*.png"), recursive=True)
norm_files = list(set([f for f in norm_files if os.path.isfile(f)]))[:500]
for f in norm_files:
    records.append({"filepath": f, "label": "Normal", "target": 0})

df = pd.DataFrame(records)
print(f"\\n📊 Total collected samples: {len(df)}")
print(df["label"].value_counts())

# Stratified Split: 70% Train, 15% Validation, 15% Test
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["target"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["target"], random_state=42)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"\\nSplits: Train = {len(train_df)} | Val = {len(val_df)} | Test = {len(test_df)}")
""")

    # =========================================================================
    # CELL 4: Data Augmentation, Dataset & Imbalance Handling
    # =========================================================================
    add_code("""# ==============================================================================
# Cell 3: Data Augmentation & Class Imbalance Handling
# ==============================================================================
# Augmentations: Chest X-rays have fixed top-to-bottom human anatomy (no vertical flips!)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=7),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class XRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["target"])
        if self.transform:
            image = self.transform(image)
        return image, label, row["filepath"]

train_dataset = XRayDataset(train_df, transform=train_transforms)
val_dataset = XRayDataset(val_df, transform=val_transforms)
test_dataset = XRayDataset(test_df, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Class Imbalance Handling: Compute inverse-frequency class weights for the loss function
class_counts = train_df["target"].value_counts().sort_index().values
total_samples = len(train_df)
class_weights = total_samples / (len(CLASS_NAMES) * class_counts.astype(np.float32))
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print("⚖️ Class weights computed for imbalance handling:")
for name, w in zip(CLASS_NAMES, class_weights):
    print(f"   {name:<14}: {w:.3f}")
""")

    # =========================================================================
    # CELL 5: Model Definition (DenseNet-121)
    # =========================================================================
    add_code("""# ==============================================================================
# Cell 4: Model Architecture (DenseNet-121 / CheXNet Standard)
# ==============================================================================
# DenseNet-121 is the gold standard for chest radiography (Rajpurkar et al., CheXNet)
def build_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    # Replace final classification head for our 3 classes
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, len(CLASS_NAMES))
    )
    return model

model = build_model().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

print("✅ DenseNet-121 loaded successfully with 3 output classes (Normal, Pneumonia, Tuberculosis).")
""")

    # =========================================================================
    # CELL 6: Train or Load from Google Drive Checkpoint
    # =========================================================================
    add_code("""# ==============================================================================
# Cell 5: Train Model OR Load from Google Drive Checkpoint
# ==============================================================================
# If the model was already trained and saved in your Google Drive, we simply LOAD it
# so you don't waste time retraining every time the session reconnects or refreshes!

EPOCHS = 5

if os.path.exists(MODEL_SAVE_PATH):
    print("=" * 70)
    print(f"💾 Found saved model in Google Drive: {MODEL_SAVE_PATH}")
    print("Loading weights directly from Google Drive... (SKIPPING TRAINING)")
    print("=" * 70)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded and ready for evaluation!")
else:
    print("=" * 70)
    print(f"⚡ No saved model found at: {MODEL_SAVE_PATH}")
    print(f"Starting training for {EPOCHS} epochs...")
    print("=" * 70)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_acc = correct / total
        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss/total:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Save best model to Google Drive
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   ⭐ Best model checkpoint saved to Google Drive: {MODEL_SAVE_PATH} ({val_acc*100:.2f}%)")
            
    print(f"\\n🎉 Training completed! Best model saved to: {MODEL_SAVE_PATH}")
""")

    # =========================================================================
    # CELL 7: Model Calibration (Temperature Scaling & ECE)
    # =========================================================================
    add_code("""# ==============================================================================
# Cell 6: Model Calibration (Temperature Scaling & Expected Calibration Error)
# ==============================================================================
# Temperature scaling adjusts output probabilities so model confidence matches real accuracy
class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_loader):
        self.model.eval()
        nll = nn.CrossEntropyLoss().to(device)
        logits_list, labels_list = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                logits_list.append(self.model(images.to(device)))
                labels_list.append(labels.to(device))
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def eval_loss():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval_loss)
        return self.temperature.item()

calibrator = TemperatureScaling(model).to(device)
optimal_temp = calibrator.fit(val_loader)
print(f"🌡️ Optimal Temperature Parameter (T): {optimal_temp:.3f}")

# Extract test set predictions
model.eval()
test_logits, test_labels = [], []
with torch.no_grad():
    for images, labels, _ in test_loader:
        test_logits.append(model(images.to(device)).cpu())
        test_labels.append(labels.numpy())

test_logits = torch.cat(test_logits)
test_labels = np.concatenate(test_labels)

uncalibrated_probs = F.softmax(test_logits, dim=1).numpy()
calibrated_probs = F.softmax(test_logits / optimal_temp, dim=1).numpy()

# Expected Calibration Error (ECE)
def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            ece += np.abs(np.mean(confidences[in_bin]) - np.mean(accuracies[in_bin])) * (np.sum(in_bin) / len(labels))
    return ece

ece_before = compute_ece(uncalibrated_probs, test_labels)
ece_after = compute_ece(calibrated_probs, test_labels)

print(f"   Uncalibrated Test ECE: {ece_before*100:.2f}%")
print(f"   Calibrated Test ECE:   {ece_after*100:.2f}% (Improvement: {(ece_before - ece_after)*100:+.2f}%)")
""")

    # =========================================================================
    # CELL 8: Evaluation & ROC Analysis
    # =========================================================================
    add_code("""# ==============================================================================
# Cell 7: Model Evaluation (Confusion Matrix & ROC Curves)
# ==============================================================================
from sklearn.preprocessing import label_binarize

test_preds = np.argmax(calibrated_probs, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title("Test Confusion Matrix", fontweight="bold")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Ground Truth")

# 2. Multi-class ROC Curves
colors = ['#2ecc71', '#e74c3c', '#f39c12']
y_test_bin = label_binarize(test_labels, classes=[0, 1, 2])

for i, (name, col) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], calibrated_probs[:, i])
    auc_score = roc_auc_score(y_test_bin[:, i], calibrated_probs[:, i])
    axes[1].plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUROC = {auc_score:.3f})")

axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
axes[1].set_title("One-vs-Rest ROC Curves", fontweight="bold")
axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
axes[1].set_ylabel("True Positive Rate (Sensitivity)")
axes[1].legend(loc="lower right")
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

print("\\n📋 Classification Report:")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4))
""")

    # =========================================================================
    # CELL 9: Horizontal Grad-CAM Saliency Maps & Sample Inspection
    # =========================================================================
    add_code("""# ==============================================================================
# Cell 8: Grad-CAM Explainability: Horizontal Image Visualization
# ==============================================================================
# Shows: [Original Chest X-Ray]  -->  [Grad-CAM Heatmap]  -->  [Superimposed Overlay]
# across randomly sampled test cases to show WHERE the model is looking!

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        
        probs = F.softmax(output / optimal_temp, dim=1)[0].detach().cpu().numpy()
        return cam, target_class, probs

# Target the final convolutional layer of DenseNet-121
target_conv_layer = model.features.denseblock4.denselayer16.conv2
grad_cam = GradCAM(model, target_conv_layer)

def make_overlay(pil_img, cam, alpha=0.45):
    img_np = np.array(pil_img.convert("RGB").resize((224, 224)))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = np.uint8(alpha * heatmap_color + (1 - alpha) * img_np)
    return img_np, heatmap_color, blended

# Pick 3 representative test samples (one Normal, one Pneumonia, one Tuberculosis)
sample_indices = []
for c in range(3):
    matches = np.where(test_labels == c)[0]
    sample_indices.append(random.choice(matches))

# Display each sample in a clean horizontal layout:
# [Original X-Ray]  -->  [Grad-CAM Heatmap]  -->  [Overlay on X-Ray]
fig, axes = plt.subplots(len(sample_indices), 3, figsize=(14, 4.2 * len(sample_indices)))

for row, idx in enumerate(sample_indices):
    img_tensor, true_label, file_path = test_dataset[idx]
    pil_img = Image.open(file_path).convert("RGB")
    
    cam, pred_class, probs = grad_cam.generate(img_tensor.unsqueeze(0).to(device))
    orig_np, heatmap_color, blended = make_overlay(pil_img, cam)
    
    # Calculate predictive entropy (uncertainty score)
    entropy = -np.sum(probs * np.log2(np.clip(probs, 1e-9, 1.0)))
    conf = probs[pred_class] * 100
    is_correct = (pred_class == true_label)
    status_color = "#27ae60" if is_correct else "#e74c3c"
    
    # Column 1: Original Image
    axes[row, 0].imshow(orig_np)
    axes[row, 0].set_title(f"Case #{idx} | Original X-Ray\\nTrue Label: {CLASS_NAMES[true_label]}", fontweight="bold")
    axes[row, 0].axis("off")
    
    # Column 2: Grad-CAM Saliency Heatmap
    axes[row, 1].imshow(cam, cmap="jet")
    axes[row, 1].set_title(f"Grad-CAM Attention Map\\n(Where the model looked)", fontweight="bold")
    axes[row, 1].axis("off")
    
    # Column 3: Overlay on X-Ray
    axes[row, 2].imshow(blended)
    axes[row, 2].set_title(f"Explainability Overlay\\nPred: {CLASS_NAMES[pred_class]} ({conf:.1f}%) | Uncertainty: {entropy:.2f} bits",
                           fontweight="bold", color=status_color)
    axes[row, 2].axis("off")

plt.suptitle("Explainable AI: Radiographic Lesion Localization via Grad-CAM", fontsize=15, fontweight="bold", y=0.99)
plt.tight_layout()
plt.show()

print("✅ Analysis complete! All evaluation metrics, calibration, and Grad-CAM visualizations generated.")
""")

    # Build Jupyter Notebook JSON
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    output_path = "Chest_XRay_Screening_Assistant_Explainability.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"🎉 Updated simplified notebook at: {os.path.abspath(output_path)}")
    print(f"   Total Cells: {len(cells)} (Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}, Code: {sum(1 for c in cells if c['cell_type'] == 'code')})")

if __name__ == "__main__":
    create_simplified_notebook()
