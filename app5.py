import streamlit as st
import cv2
import os
import time
import math
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
st.set_page_config(
    page_title="Minecraft Super Resolution Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. MODEL ARCHITECTURE (Matches latest train.py)
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
        assert scale > 0 and (scale & (scale - 1)) == 0, f"Scale must be power of 2, got {scale}"
        num_stages = int(math.log2(scale))
        layers = []
        for _ in range(num_stages):
            layers.extend([
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
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
        return x

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return torch.clamp(self.forward(x), 0.0, 1.0)


# ============================================================
# 3. MODEL LOADER
# ============================================================
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinecraftSR(num_features=64, num_blocks=8, scale=4).to(device)

    weights_path = "minecraft_ai_x4_best.pth"
    if not os.path.exists(weights_path):
        weights_path = "minecraft_ai_x4_latest.pth"

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, device, weights_path
    return None, device, None


model, device, loaded_weights = load_trained_model()

# ============================================================
# 4. SIDEBAR SETUP
# ============================================================
st.sidebar.title("Controls & Settings")

try:
    available_images = sorted([
        f for f in os.listdir("HR")
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ])
except FileNotFoundError:
    available_images = []

if not available_images:
    st.sidebar.error("No images found in HR directory.")
    selected_image = None
else:
    selected_image = st.sidebar.selectbox("Test Sequence Image", available_images)

baseline_methods = {
    "Bicubic Interpolation": cv2.INTER_CUBIC,
    "Bilinear Interpolation": cv2.INTER_LINEAR,
    "Nearest Neighbor": cv2.INTER_NEAREST
}
selected_baseline = st.sidebar.selectbox("Baseline Comparison", list(baseline_methods.keys()))
baseline_flag = baseline_methods[selected_baseline]

comparison_target = st.sidebar.radio(
    "Slider Left Image:",
    ["Trained AI (MinecraftSR)", f"Baseline ({selected_baseline.split()[0]})"]
)

if loaded_weights:
    st.sidebar.success(f"Loaded: `{loaded_weights}` ({device})")
else:
    st.sidebar.warning("No checkpoint found (`minecraft_ai_x4_best.pth`). Using baselines only.")

# ============================================================
# 5. INFERENCE & METRIC PIPELINE
# ============================================================
@st.cache_data
def evaluate_frame(image_name, baseline_flag_val):
    hr_path = os.path.join("HR", image_name)
    lr_path = os.path.join("LR", image_name)

    hr_img = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)
    lr_img = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
    h, w = hr_img.shape[:2]

    # Baseline computation
    baseline_img = cv2.resize(lr_img, (w, h), interpolation=baseline_flag_val)
    b_psnr = compute_psnr(hr_img, baseline_img)
    b_ssim = compute_ssim(hr_img, baseline_img, channel_axis=2)

    ai_img = baseline_img.copy()
    a_psnr, a_ssim, latency_ms = 0.0, 0.0, 0.0

    if model is not None:
        # Prepare LR tensor: (B, C, H, W) normalized to [0, 1]
        lr_tensor = torch.from_numpy(lr_img.transpose((2, 0, 1))).float() / 255.0
        lr_tensor = lr_tensor.unsqueeze(0).to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output_tensor = model.predict(lr_tensor)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        out_np = output_tensor.squeeze(0).cpu().numpy()
        ai_img = (out_np.transpose((1, 2, 0)) * 255.0).astype(np.uint8)

        a_psnr = compute_psnr(hr_img, ai_img)
        a_ssim = compute_ssim(hr_img, ai_img, channel_axis=2)

    return hr_img, lr_img, baseline_img, ai_img, b_psnr, b_ssim, a_psnr, a_ssim, latency_ms


# ============================================================
# 6. RENDER DASHBOARD
# ============================================================
st.title("Spatio-Temporal Super Resolution Engine")
st.markdown("Evaluating Sobel-trained **MinecraftSR** vs. Classical Interpolation Baselines.")

if selected_image:
    hr_img, lr_img, baseline_img, ai_img, b_psnr, b_ssim, a_psnr, a_ssim, latency = evaluate_frame(
        selected_image, baseline_flag
    )

    st.subheader("Quantitative Evaluation")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scale Factor", f"{lr_img.shape[1]}x{lr_img.shape[0]} → {hr_img.shape[1]}x{hr_img.shape[0]}")
    c2.metric(f"{selected_baseline.split()[0]} PSNR", f"{b_psnr:.2f} dB")
    c3.metric(f"{selected_baseline.split()[0]} SSIM", f"{b_ssim:.4f}")

    if loaded_weights:
        c4.metric("AI PSNR", f"{a_psnr:.2f} dB", delta=f"{a_psnr - b_psnr:+.2f} dB")
        c5.metric("AI SSIM", f"{a_ssim:.4f}", delta=f"{a_ssim - b_ssim:+.4f}")
    else:
        c4.metric("AI PSNR", "N/A")
        c5.metric("AI SSIM", "N/A")

    st.markdown("---")
    st.subheader("Visual Analysis")

    if comparison_target.startswith("Trained AI") and loaded_weights:
        left_display = Image.fromarray(ai_img)
        left_label = f"MinecraftSR (Edge-Loss) [{latency:.1f} ms]"
    else:
        left_display = Image.fromarray(baseline_img)
        left_label = f"{selected_baseline.split()[0]} Baseline"

    image_comparison(
        img1=left_display,
        img2=Image.fromarray(hr_img),
        label1=left_label,
        label2="Native 1080p Ground Truth",
        width=1000,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )