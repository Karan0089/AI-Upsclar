import streamlit as st
import cv2
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from streamlit_image_comparison import image_comparison
from PIL import Image
import pandas as pd

from temporal_warp import compute_dense_optical_flow, warp_frame_torch

# ============================================================
# PAGE CONFIGURATION & METADATA
# ============================================================
st.set_page_config(
    page_title="Game Texture Super-Resolution Engine",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# NEURAL ARCHITECTURES
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
        assert scale > 0 and (scale & (scale - 1)) == 0
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
    """Stage 2: 3-Channel Spatial-Only Sub-Pixel Residual Network"""
    def __init__(self, in_channels=3, num_features=64, num_blocks=8, scale=4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
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
        return torch.clamp(self.tail(x), 0.0, 1.0)

class MinecraftTemporalSR(nn.Module):
    """Stage 3: 6-Channel Motion-Compensated Spatio-Temporal Residual Network"""
    def __init__(self, in_channels=6, num_features=64, num_blocks=8, scale=4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
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
        return torch.clamp(self.tail(x), 0.0, 1.0)

# ============================================================
# CACHED MODEL CHECKPOINT LOADER
# ============================================================
@st.cache_resource
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Spatial Model
    spatial_model = MinecraftSR(in_channels=3, num_features=64, num_blocks=8, scale=4).to(device)
    spatial_path = "minecraft_ai_x4_best.pth"
    spatial_loaded = False
    if os.path.exists(spatial_path):
        ckpt = torch.load(spatial_path, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        spatial_model.load_state_dict(state)
        spatial_model.eval()
        spatial_loaded = True

    # Load Temporal Model
    temporal_model = MinecraftTemporalSR(in_channels=6, num_features=64, num_blocks=8, scale=4).to(device)
    temporal_path = "minecraft_temporal_seq_x4_best.pth"
    if not os.path.exists(temporal_path):
        temporal_path = "minecraft_temporal_x4_best.pth"
    temporal_loaded = False
    if os.path.exists(temporal_path):
        ckpt = torch.load(temporal_path, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        temporal_model.load_state_dict(state)
        temporal_model.eval()
        temporal_loaded = True

    return spatial_model, temporal_model, device, spatial_loaded, temporal_loaded

spatial_net, temporal_net, device, has_spatial, has_temporal = load_all_models()

# ============================================================
# UTILITIES: METRICS & ERROR HEATMAPS
# ============================================================
def generate_error_heatmap(target_rgb, pred_rgb):
    diff = np.abs(target_rgb.astype(np.float32) - pred_rgb.astype(np.float32)).mean(axis=2)
    diff_norm = np.clip(diff / 50.0 * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

def calculate_temporal_warp_error(prev_rgb, curr_rgb):
    """
    Computes inter-frame motion-compensated error:
    E_warp = || I_t - Warp(I_{t-1}) ||_1
    Lower values indicate consistent frame transitions and lower visual flicker.
    """
    flow = compute_dense_optical_flow(prev_rgb, curr_rgb)
    prev_t = torch.from_numpy(prev_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    flow_t = torch.from_numpy(flow).float().unsqueeze(0)
    warped_prev = warp_frame_torch(prev_t, flow_t).squeeze(0).numpy().transpose(1, 2, 0) * 255.0
    return float(np.mean(np.abs(curr_rgb.astype(np.float32) - warped_prev)))

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.title("🎮 Pipeline Architecture")

stage_mode = st.sidebar.radio(
    "Development Stage:",
    [
        "Stage 1: Classical Baselines (Math Only)",
        "Stage 2: Spatial Neural Upscaling (Sobel L1)",
        "Stage 3: Spatio-Temporal Fusion (6-Channel)",
        "Stage 4: Full Ablation & Sequence Benchmark"
    ]
)

# Dataset directory selector
dataset_source = "seq" if ("Stage 3" in stage_mode or "Stage 4" in stage_mode) else "any"
if os.path.exists("LR_seq") and len(os.listdir("LR_seq")) > 1:
    lr_folder = "LR_seq"
    hr_folder = "HR_seq"
else:
    lr_folder = "LR"
    hr_folder = "HR"

try:
    frames = sorted([f for f in os.listdir(lr_folder) if f.lower().endswith(('.png', '.jpg'))])
except FileNotFoundError:
    frames = []

with st.sidebar.expander("⚙️ Target Frame Selection", expanded=True):
    if len(frames) >= 2:
        frame_idx = st.slider("Sequence Index (t)", min_value=1, max_value=len(frames) - 1, value=1)
        prev_frame_name = frames[frame_idx - 1]
        curr_frame_name = frames[frame_idx]
        st.caption(f"Active pair: `{prev_frame_name}` (t-1) → `{curr_frame_name}` (t)")
    else:
        st.error("Missing dataset frames in directory.")
        st.stop()

with st.sidebar.expander("🔍 ROI Zoom Inspector"):
    enable_zoom = st.checkbox("Enable Region-of-Interest Crop", value=False)
    crop_size = st.slider("Crop Box Size (HR Pixels)", 128, 512, 256, step=64) if enable_zoom else 0

# Checkpoint status badges
st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Loaded Weights")
st.sidebar.markdown(f"- **Spatial Net:** `{'Loaded' if has_spatial else 'Not Found'}`")
st.sidebar.markdown(f"- **Temporal Net:** `{'Loaded' if has_temporal else 'Not Found'}`")
st.sidebar.markdown(f"- **Compute Device:** `{device}`")

# ============================================================
# DATA PIPELINE EXECUTION
# ============================================================
lr_prev_bgr = cv2.imread(os.path.join(lr_folder, prev_frame_name))
lr_curr_bgr = cv2.imread(os.path.join(lr_folder, curr_frame_name))
hr_curr_bgr = cv2.imread(os.path.join(hr_folder, curr_frame_name))

lr_curr_rgb = cv2.cvtColor(lr_curr_bgr, cv2.COLOR_BGR2RGB)
hr_curr_rgb = cv2.cvtColor(hr_curr_bgr, cv2.COLOR_BGR2RGB)
h_hr, w_hr = hr_curr_rgb.shape[:2]

# Baselines
nearest_rgb = cv2.resize(lr_curr_rgb, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)
bilinear_rgb = cv2.resize(lr_curr_rgb, (w_hr, h_hr), interpolation=cv2.INTER_LINEAR)
bicubic_rgb = cv2.resize(lr_curr_rgb, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)

# Spatial AI Inference
spatial_rgb = bicubic_rgb.copy()
spatial_lat = 0.0
if has_spatial:
    t0 = time.perf_counter()
    lr_t = torch.from_numpy(lr_curr_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        out_t = spatial_net(lr_t)
    spatial_lat = (time.perf_counter() - t0) * 1000.0
    spatial_rgb = (out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

# Temporal AI Inference
temporal_rgb = bicubic_rgb.copy()
temporal_lat = 0.0
if has_temporal:
    t0 = time.perf_counter()
    flow = compute_dense_optical_flow(lr_prev_bgr, lr_curr_bgr)
    prev_lr_t = torch.from_numpy(lr_prev_bgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    flow_t = torch.from_numpy(flow).float().unsqueeze(0)
    warped_prev_t = warp_frame_torch(prev_lr_t, flow_t).to(device)
    curr_lr_t = (torch.from_numpy(lr_curr_bgr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0).to(device)
    input_6ch = torch.cat([curr_lr_t, warped_prev_t], dim=1)
    
    with torch.no_grad():
        out_temp = temporal_net(input_6ch)
    temporal_lat = (time.perf_counter() - t0) * 1000.0
    temp_np = out_temp.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    temporal_rgb = cv2.cvtColor((temp_np * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)

# Apply ROI Crop if requested
if enable_zoom:
    cy, cx = h_hr // 2, w_hr // 2
    hs = crop_size // 2
    y1, y2 = max(0, cy - hs), min(h_hr, cy + hs)
    x1, x2 = max(0, cx - hs), min(w_hr, cx + hs)
    
    hr_curr_rgb = hr_curr_rgb[y1:y2, x1:x2]
    nearest_rgb = nearest_rgb[y1:y2, x1:x2]
    bicubic_rgb = bicubic_rgb[y1:y2, x1:x2]
    spatial_rgb = spatial_rgb[y1:y2, x1:x2]
    temporal_rgb = temporal_rgb[y1:y2, x1:x2]

# Compute Primary Metrics
psnr_bicubic = compute_psnr(hr_curr_rgb, bicubic_rgb)
ssim_bicubic = compute_ssim(hr_curr_rgb, bicubic_rgb, channel_axis=2)

psnr_spatial = compute_psnr(hr_curr_rgb, spatial_rgb)
ssim_spatial = compute_ssim(hr_curr_rgb, spatial_rgb, channel_axis=2)

psnr_temporal = compute_psnr(hr_curr_rgb, temporal_rgb)
ssim_temporal = compute_ssim(hr_curr_rgb, temporal_rgb, channel_axis=2)

# ============================================================
# STAGE ROUTING & RENDERING
# ============================================================
st.title("Game Texture Super-Resolution Engine")

if "Stage 1" in stage_mode:
    st.subheader("Stage 1: Traditional Mathematical Interpolation Baselines")
    st.caption("Evaluating fundamental limiters of non-learned upsampling: Nearest Neighbor (aliased) vs. Bicubic (blurred).")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scale Factor", f"4x ({w_hr//4}x{h_hr//4} → {w_hr}x{h_hr})")
    c2.metric("Nearest PSNR", f"{compute_psnr(hr_curr_rgb, nearest_rgb):.2f} dB")
    c3.metric("Bicubic PSNR", f"{psnr_bicubic:.2f} dB")
    c4.metric("Bicubic SSIM", f"{ssim_bicubic:.4f}")

    image_comparison(
        img1=Image.fromarray(bicubic_rgb),
        img2=Image.fromarray(hr_curr_rgb),
        label1="Classical Bicubic",
        label2="Ground Truth 1080p",
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )

elif "Stage 2" in stage_mode:
    st.subheader("Stage 2: Single-Frame Sub-Pixel Convolution (Spatial-Only)")
    st.caption("Architecture: 8 Residual Blocks + 2-Stage PixelShuffle | Loss: L1 + Sobel Edge Gradient Loss.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inference Latency", f"{spatial_lat:.1f} ms")
    c2.metric("Bicubic PSNR", f"{psnr_bicubic:.2f} dB")
    c3.metric("Spatial AI PSNR", f"{psnr_spatial:.2f} dB", delta=f"{psnr_spatial - psnr_bicubic:+.2f} dB")
    c4.metric("Spatial AI SSIM", f"{ssim_spatial:.4f}", delta=f"{ssim_spatial - ssim_bicubic:+.4f}")

    col_view, col_heat = st.columns([3, 2])
    with col_view:
        image_comparison(
            img1=Image.fromarray(spatial_rgb),
            img2=Image.fromarray(hr_curr_rgb),
            label1="MinecraftSR (Spatial)",
            label2="Ground Truth 1080p",
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )
    with col_heat:
        st.markdown("**Spatial Reconstruction Error Heatmap**")
        st.image(generate_error_heatmap(hr_curr_rgb, spatial_rgb), use_column_width=True, caption="Blue = Perfect Match, Red = High Residual Error")

elif "Stage 3" in stage_mode:
    st.subheader("Stage 3: Multi-Frame Spatio-Temporal Accumulation")
    st.caption("Architecture: 6-Channel Input [RGB(t) + Warped RGB(t-1)] | Alignment: Gunnar Farnebäck Optical Flow.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Temporal Latency", f"{temporal_lat:.1f} ms")
    c2.metric("Spatial PSNR", f"{psnr_spatial:.2f} dB")
    c3.metric("Temporal PSNR", f"{psnr_temporal:.2f} dB", delta=f"{psnr_temporal - psnr_spatial:+.2f} dB")
    c4.metric("Temporal SSIM", f"{ssim_temporal:.4f}", delta=f"{ssim_temporal - ssim_spatial:+.4f}")
    c5.metric("Net Gain vs Bicubic", f"{psnr_temporal - psnr_bicubic:+.2f} dB")

    image_comparison(
        img1=Image.fromarray(spatial_rgb),
        img2=Image.fromarray(temporal_rgb),
        label1="Spatial AI (Single Frame)",
        label2="Spatio-Temporal AI (Multi-Frame)",
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )

elif "Stage 4" in stage_mode:
    st.subheader("Stage 4: Full Ablation, Flicker Analysis & Sequence Benchmark")
    st.caption("Comprehensive comparative benchmark: Mathematical baselines vs. Single-Frame AI vs. Multi-Frame Temporal Fusion.")

    # Quantitative ablation table
    df_ablation = pd.DataFrame({
        "Upscaling Pipeline": ["Bicubic Interpolation", "MinecraftSR (Spatial Only)", "MinecraftTemporalSR (Spatio-Temporal)"],
        "PSNR (dB)": [f"{psnr_bicubic:.2f}", f"{psnr_spatial:.2f}", f"{psnr_temporal:.2f}"],
        "SSIM": [f"{ssim_bicubic:.4f}", f"{ssim_spatial:.4f}", f"{ssim_temporal:.4f}"],
        "Δ PSNR vs Base": ["0.00 dB", f"{psnr_spatial - psnr_bicubic:+.2f} dB", f"{psnr_temporal - psnr_bicubic:+.2f} dB"],
        "Latency (RTX 3060)": ["< 1.0 ms", f"{spatial_lat:.1f} ms", f"{temporal_lat:.1f} ms"]
    })
    st.dataframe(df_ablation, use_container_width=True)

    # 3-Way Diagnostic Heatmap Row
    st.markdown("### Residual Error Heatmap Breakdown")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("**Bicubic Residual Error**")
        st.image(generate_error_heatmap(hr_curr_rgb, bicubic_rgb), use_column_width=True)
    with h2:
        st.markdown("**Spatial Model Error**")
        st.image(generate_error_heatmap(hr_curr_rgb, spatial_rgb), use_column_width=True)
    with h3:
        st.markdown("**Temporal Model Error**")
        st.image(generate_error_heatmap(hr_curr_rgb, temporal_rgb), use_column_width=True)

    # Multi-frame sequence comparison slider
    st.markdown("### Side-by-Side Reconstruction Inspection")
    target_ablation = st.selectbox("Select Model to Compare with Ground Truth 1080p:", [
        "Spatio-Temporal Model (Stage 3)",
        "Spatial-Only Model (Stage 2)",
        "Bicubic Baseline (Stage 1)"
    ])
    
    active_comp = temporal_rgb if "Spatio-Temporal" in target_ablation else (spatial_rgb if "Spatial-Only" in target_ablation else bicubic_rgb)
    image_comparison(
        img1=Image.fromarray(active_comp),
        img2=Image.fromarray(hr_curr_rgb),
        label1=target_ablation.split()[0],
        label2="Native Ground Truth",
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )