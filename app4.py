import streamlit as st
import cv2
import os
import time
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from streamlit_image_comparison import image_comparison
from PIL import Image

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="AI Super Resolution Evaluator", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# 2. MODEL ARCHITECTURE (Exact Match to train.py)
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale=4):
        super().__init__()
        layers = []
        if scale == 4:
            for _ in range(2):
                layers.append(nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(2))
                layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MinecraftSR(nn.Module):
    def __init__(self, num_features=64, num_blocks=8, scale=4):
        super().__init__()
        self.head = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsample = UpsampleBlock(num_features, scale)
        self.tail = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        skip = x
        x = self.body(x)
        x = self.body_conv(x)
        x = x + skip
        x = self.upsample(x)
        x = self.tail(x)
        return torch.clamp(x, 0.0, 1.0)


# ============================================================
# 3. CACHED MODEL LOADER
# ============================================================
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinecraftSR(num_features=64, num_blocks=8, scale=4).to(device)
    
    weights_path = "minecraft_ai_x4_best.pth"
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        # Check if saved checkpoint is full dict or just state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, device, True
    else:
        return None, device, False

model, device, model_loaded = load_trained_model()


# ============================================================
# 4. SIDEBAR CONTROLS
# ============================================================
st.sidebar.title("Pipeline Controls")

# Image Selector
try:
    available_images = sorted([f for f in os.listdir("HR") if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
except FileNotFoundError:
    available_images = []

if not available_images:
    st.sidebar.error("No images found in HR directory.")
    selected_image = None
else:
    selected_image = st.sidebar.selectbox("Test Sequence Frame", available_images)

# Baseline Selector
baseline_methods = {
    "Bicubic Interpolation": cv2.INTER_CUBIC,
    "Bilinear Interpolation": cv2.INTER_LINEAR,
    "Nearest Neighbor": cv2.INTER_NEAREST
}
selected_baseline = st.sidebar.selectbox("Baseline Algorithm", list(baseline_methods.keys()))
baseline_flag = baseline_methods[selected_baseline]

# Slider Left Side Target
view_mode = st.sidebar.radio("Left Slider Displays:", ["Trained AI (MinecraftSR)", f"Baseline ({selected_baseline})"])

if model_loaded:
    st.sidebar.success(f"Loaded weights: minecraft_ai_x4_best.pth ({device})")
else:
    st.sidebar.warning("Model weights not found. Running baseline comparisons only.")


# ============================================================
# 5. INFERENCE & PROCESSING PIPELINE
# ============================================================
@st.cache_data
def run_evaluation(image_name, method_flag):
    hr_path = os.path.join("HR", image_name)
    lr_path = os.path.join("LR", image_name)
    
    hr_img = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)
    lr_img = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
    h, w = hr_img.shape[:2]
    
    # 1. Baseline interpolation
    baseline_img = cv2.resize(lr_img, (w, h), interpolation=method_flag)
    base_psnr = compute_psnr(hr_img, baseline_img)
    base_ssim = compute_ssim(hr_img, baseline_img, channel_axis=2)
    
    # 2. AI Inference
    ai_img = baseline_img.copy()
    ai_psnr, ai_ssim, latency_ms = 0.0, 0.0, 0.0
    
    if model_loaded and model is not None:
        # Prepare LR tensor: (C, H, W) normalized to [0, 1]
        lr_tensor = torch.from_numpy(lr_img.transpose((2, 0, 1))).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(device)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            output_tensor = model(lr_tensor)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Convert tensor back to image array
        out_np = output_tensor.squeeze(0).cpu().numpy()
        out_np = np.clip(out_np, 0.0, 1.0)
        ai_img = (out_np.transpose((1, 2, 0)) * 255.0).astype(np.uint8)
        
        ai_psnr = compute_psnr(hr_img, ai_img)
        ai_ssim = compute_ssim(hr_img, ai_img, channel_axis=2)
        
    return hr_img, lr_img, baseline_img, ai_img, base_psnr, base_ssim, ai_psnr, ai_ssim, latency_ms


# ============================================================
# 6. DASHBOARD RENDERING
# ============================================================
st.title("Spatio-Temporal Super Resolution Engine")
st.markdown("Comparing traditional mathematical interpolation vs. our custom sub-pixel residual network.")

if selected_image:
    hr_img, lr_img, baseline_img, ai_img, b_psnr, b_ssim, a_psnr, a_ssim, latency = run_evaluation(selected_image, baseline_flag)
    
    # Metrics
    st.subheader("Quantitative Evaluation")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Resolution", f"{lr_img.shape[1]}x{lr_img.shape[0]} → {hr_img.shape[1]}x{hr_img.shape[0]}")
    c2.metric(f"{selected_baseline.split()[0]} PSNR", f"{b_psnr:.2f} dB")
    c3.metric(f"{selected_baseline.split()[0]} SSIM", f"{b_ssim:.4f}")
    
    if model_loaded:
        psnr_delta = f"{a_psnr - b_psnr:+.2f} dB"
        ssim_delta = f"{a_ssim - b_ssim:+.4f}"
        c4.metric("AI PSNR", f"{a_psnr:.2f} dB", delta=psnr_delta)
        c5.metric("AI SSIM", f"{a_ssim:.4f}", delta=ssim_delta)
    else:
        c4.metric("AI PSNR", "N/A")
        c5.metric("AI SSIM", "N/A")

    st.markdown("---")
    
    # Visual Interactive Slider
    st.subheader("Visual Analysis")
    if view_mode.startswith("Trained AI") and model_loaded:
        left_img = Image.fromarray(ai_img)
        left_label = f"AI Reconstruction ({latency:.1f} ms)"
    else:
        left_img = Image.fromarray(baseline_img)
        left_label = f"Baseline ({selected_baseline.split()[0]})"

    image_comparison(
        img1=left_img,
        img2=Image.fromarray(hr_img),
        label1=left_label,
        label2="Native 1080p Ground Truth",
        width=1000,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )