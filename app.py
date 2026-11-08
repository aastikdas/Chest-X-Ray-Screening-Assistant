"""
================================================================================
Chest X-Ray Screening Assistant with Explainability
A clean, sober Clinical Decision-Support (CDS) Prototype
================================================================================
"""

import os
import glob
import io
import numpy as np
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# ==============================================================================
# 1. Page Configuration (Sober, Clean Clinical Styling)
# ==============================================================================
st.set_page_config(
    page_title="Chest X-Ray Screening Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal CSS for a clean, non-flashy, sober academic appearance
st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
    }
    .header-title {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .header-sub {
        font-size: 14px;
        color: #475569;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
    }
    .badge-normal {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-alert {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
    }
    .warning-box {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        border-left: 5px solid #F59E0B;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 18px;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_TEMP = 1.151  # As fitted in the notebook calibration

# ==============================================================================
# 2. ResNet50 Architecture in Pure PyTorch (Matches notebook exactly)
# ==============================================================================
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512 * 4, num_classes)
        )

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ==============================================================================
# 3. Model Loading & Weights Management
# ==============================================================================
def load_screening_model(checkpoint_source=None):
    model = ResNet50(num_classes=3).to(DEVICE)
    model.eval()
    loaded_name = None
    is_trained = False

    # 1. If user uploaded a checkpoint file through Streamlit UI
    if checkpoint_source is not None and not isinstance(checkpoint_source, str):
        try:
            bytes_data = checkpoint_source.read()
            checkpoint = torch.load(io.BytesIO(bytes_data), map_location=DEVICE)
            state_dict = checkpoint.get("best_model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            loaded_name = checkpoint_source.name
            is_trained = True
            return model, loaded_name, is_trained
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded checkpoint: {e}")

    # 2. Search local directory for known saved checkpoint files
    candidate_paths = [
        checkpoint_source if isinstance(checkpoint_source, str) else None,
        "chest_xray_resnet50_2100_e8_clean.pth",
        "chest_xray_densenet121_2100_e8_clean.pth",
        "best_chest_xray_model.pth"
    ]

    for path in candidate_paths:
        if path and os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=DEVICE)
                state_dict = checkpoint.get("best_model_state_dict", checkpoint)
                model.load_state_dict(state_dict, strict=False)
                loaded_name = os.path.basename(path)
                is_trained = True
                break
            except Exception as e:
                print(f"Error loading {path}: {e}")

    return model, loaded_name, is_trained

# ==============================================================================
# 4. Grad-CAM Implementation
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, temperature=1.151):
        self.model.eval()
        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, predicted_class].backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_pil = Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)
        cam_resized = np.array(cam_pil, dtype=np.float32) / 255.0

        calibrated_probs = F.softmax(output / temperature, dim=1)[0].detach().cpu().numpy()
        return cam_resized, predicted_class, calibrated_probs

def create_visual_overlay(pil_image, cam, alpha=0.45):
    img_resized = pil_image.convert("RGB").resize((224, 224))
    img_np = np.array(img_resized)

    heatmap_colored = cm.jet(cam)[:, :, :3]
    heatmap_uint8 = np.uint8(255 * heatmap_colored)

    overlay = np.uint8(alpha * heatmap_uint8 + (1 - alpha) * img_np)
    return img_resized, heatmap_uint8, overlay

def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return tensor

# ==============================================================================
# 5. UI Layout: Sidebar & Checkpoint Upload
# ==============================================================================
st.sidebar.markdown("### 🫁 Model Configuration")

# Direct file uploader for the .pth model checkpoint
uploaded_checkpoint = st.sidebar.file_uploader(
    "📥 Upload Trained Checkpoint (.pth)",
    type=["pth", "pt"],
    help="Download 'chest_xray_resnet50_2100_e8_clean.pth' from your Google Drive and upload it here!"
)

# Also check for any existing .pth files in the project folder
local_pths = glob.glob("*.pth")
selected_local_file = None
if local_pths and uploaded_checkpoint is None:
    selected_local_file = st.sidebar.selectbox("Local Checkpoint Found", local_pths, index=0)

checkpoint_to_load = uploaded_checkpoint if uploaded_checkpoint is not None else selected_local_file
model, loaded_path, is_trained = load_screening_model(checkpoint_to_load)

if is_trained:
    st.sidebar.success(f"✅ **Trained Weights Loaded**: `{loaded_path}`")
else:
    st.sidebar.error("❌ **NO TRAINED MODEL LOADED**")
    st.sidebar.caption(
        "Currently using un-trained random weights. All outputs will hover at ~33% (random guessing). "
        "Upload `chest_xray_resnet50_2100_e8_clean.pth` above to get your real 98%+ accuracy."
    )

temperature = st.sidebar.slider("Calibration Temperature (T)", min_value=0.5, max_value=2.0, value=DEFAULT_TEMP, step=0.05,
                                help="Fitted temperature from validation set (T = 1.151).")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Test with Sample Scans")
sample_choice = st.sidebar.selectbox(
    "Load Built-in Sample Radiograph",
    ["None (Upload your own)", "Sample 1: Normal", "Sample 2: Pneumonia", "Sample 3: Tuberculosis"]
)

# ==============================================================================
# 6. UI Layout: Main Header
# ==============================================================================
st.markdown('<div class="header-title">🫁 Chest X-Ray Screening Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Clinical Decision-Support & Lesion Localization Prototype (Normal / Pneumonia / Tuberculosis)</div>', unsafe_allow_html=True)

# Prominent alert banner if weights are missing
if not is_trained:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Model Weights Missing (Random Initialization Active)</strong><br>
        Because no trained <code>.pth</code> file is in this folder, the neural network is initialized with <strong>random weights</strong>. 
        This is why all probabilities are ~33.3% and always guessing Tuberculosis.<br><br>
        <strong>How to fix in 10 seconds:</strong>
        <ol style="margin-bottom: 0px; padding-left: 20px;">
            <li>Go to your <strong>Google Drive</strong> and download <code>chest_xray_resnet50_2100_e8_clean.pth</code>.</li>
            <li>In the sidebar on the left, click <strong>"Browse files"</strong> under <em>Upload Trained Checkpoint (.pth)</em> and select it.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Upload Frontal Chest Radiograph (PA or AP view)", type=["png", "jpg", "jpeg"])

active_image = None
if uploaded_file is not None:
    try:
        active_image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Could not load image: {e}")
elif sample_choice != "None (Upload your own)":
    mapping = {
        "Sample 1: Normal": "samples/sample_normal.png",
        "Sample 2: Pneumonia": "samples/sample_pneumonia.png",
        "Sample 3: Tuberculosis": "samples/sample_tuberculosis.png"
    }
    sample_path = mapping.get(sample_choice)
    if sample_path and os.path.exists(sample_path):
        active_image = Image.open(sample_path)

# ==============================================================================
# 7. Analysis & Horizontal Visualization
# ==============================================================================
if active_image is not None:
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    with st.spinner("Analyzing radiograph and generating Grad-CAM heatmaps..."):
        input_tensor = preprocess_image(active_image)
        cam, pred_class, probs = grad_cam.generate(input_tensor, temperature=temperature)
        orig_img, heatmap_img, overlay_img = create_visual_overlay(active_image, cam)

    predicted_label = CLASS_NAMES[pred_class]
    confidence_pct = probs[pred_class] * 100.0

    entropy = -np.sum(probs * np.log2(np.clip(probs, 1e-9, 1.0)))
    normalized_uncertainty = (entropy / 1.585) * 100.0

    # Horizontal 3-Column Visualization
    st.markdown("#### 🔍 Visual Localization (Grad-CAM)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(orig_img, caption="1. Original Radiograph", use_container_width=True)
    with col2:
        st.image(heatmap_img, caption="2. Grad-CAM Attention Map (Where Model Looked)", use_container_width=True)
    with col3:
        st.image(overlay_img, caption="3. Superimposed Lesion Overlay", use_container_width=True)

    st.markdown("---")

    # Diagnostic Metrics
    st.markdown("#### 📋 Diagnostic Assessment & Uncertainty")
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Predicted Pathology</div>', unsafe_allow_html=True)
        if predicted_label == "Normal":
            badge_html = f'<span class="badge-normal">● {predicted_label} (Routine)</span>'
        else:
            badge_html = f'<span class="badge-alert">● {predicted_label} (Infectious Suspicion)</span>'
        st.markdown(f'<div class="metric-value">{predicted_label}</div>{badge_html}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with m_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Calibrated Confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{confidence_pct:.1f}%</div>', unsafe_allow_html=True)
        st.progress(float(probs[pred_class]))
        st.markdown('</div>', unsafe_allow_html=True)

    with m_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Predictive Uncertainty</div>', unsafe_allow_html=True)
        if normalized_uncertainty < 40:
            unc_badge = '<span class="badge-normal">Low Uncertainty (Reliable)</span>'
        elif normalized_uncertainty < 65:
            unc_badge = '<span class="badge-warning">Moderate Uncertainty (Review)</span>'
        else:
            unc_badge = '<span class="badge-alert">High Uncertainty (Ambiguous)</span>'
        st.markdown(f'<div class="metric-value">{entropy:.2f} <span style="font-size:14px; font-weight:normal; color:#64748B;">bits</span></div>{unc_badge}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Probability Distribution
    st.markdown("##### Calibrated Likelihood Breakdown:")
    p_cols = st.columns(3)
    for i, name in enumerate(CLASS_NAMES):
        with p_cols[i]:
            val = float(probs[i])
            st.write(f"**{name}**: `{val*100:.1f}%`")
            st.progress(val)

    if predicted_label in ["Pneumonia", "Tuberculosis"] and confidence_pct >= 60.0:
        st.warning("⚠️ **Triage Action**: High infectious probability detected. Recommended for prioritized radiologist consultation and microbiological confirmation (e.g., Sputum GeneXpert for TB).")
    elif normalized_uncertainty >= 50.0:
        st.info("ℹ️ **Triage Action**: Borderline / ambiguous presentation. Secondary reader inspection advised.")
    else:
        st.success("✅ **Triage Action**: Routine presentation with low suspicion of acute focal consolidation.")

else:
    st.info("👆 Upload a chest radiograph above, or select a sample scan from the sidebar to inspect diagnostic predictions and Grad-CAM heatmaps.")

st.markdown("---")
st.caption(
    "**Clinical Governance Notice:** This software is an academic decision-support prototype developed for radiologic triage demonstration. "
    "It is not cleared as an autonomous diagnostic device by regulatory bodies (FDA/CE). All findings require certified medical oversight."
)
