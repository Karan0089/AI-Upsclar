import streamlit as st
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from streamlit_image_comparison import image_comparison
from PIL import Image

st.set_page_config(page_title="Upscaling Baseline Evaluator", layout="wide", initial_sidebar_state="expanded")

# --- 1. Sidebar Configuration ---
st.sidebar.title("Configuration")
st.sidebar.markdown("Select an image and a traditional upscaling method to evaluate.")

# Image Selector
try:
    available_images = sorted([f for f in os.listdir("HR") if f.endswith(('.png', '.jpg', '.jpeg'))])
except FileNotFoundError:
    available_images = []

selected_image = st.sidebar.selectbox("Test Sequence Image", available_images) if available_images else None

# Interpolation Selector (Lanczos Removed)
method_names = {
    "Nearest Neighbor (Pixelated/Sharp)": cv2.INTER_NEAREST,
    "Bilinear (Fast/Soft)": cv2.INTER_LINEAR,
    "Bicubic (Standard/Smooth)": cv2.INTER_CUBIC
}
selected_method_name = st.sidebar.selectbox("Interpolation Algorithm", list(method_names.keys()))
selected_method_flag = method_names[selected_method_name]

st.title("Pre-AI Baseline Evaluation")
st.markdown("Analyzing the structural limits of traditional mathematical upscaling algorithms.")

# --- 2. Processing Engine ---
@st.cache_data
def process_interpolation(image_name, method_flag):
    hr_path = os.path.join("HR", image_name)
    lr_path = os.path.join("LR", image_name)
    
    hr_img = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)
    lr_img = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
    
    height, width = hr_img.shape[:2]
    
    # Upscale using the user-selected mathematical method
    upscaled_img = cv2.resize(lr_img, (width, height), interpolation=method_flag)
    
    psnr_val = compute_psnr(hr_img, upscaled_img)
    ssim_val = compute_ssim(hr_img, upscaled_img, channel_axis=2)
    
    return hr_img, lr_img, upscaled_img, psnr_val, ssim_val

# --- 3. Render Dashboard ---
if selected_image:
    try:
        hr_img, lr_img, upscaled_img, current_psnr, current_ssim = process_interpolation(selected_image, selected_method_flag)
        
        st.subheader("Quantitative Evaluation")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Input (LR)", f"{lr_img.shape[1]}x{lr_img.shape[0]}")
        col2.metric("Target (HR)", f"{hr_img.shape[1]}x{hr_img.shape[0]}")
        col3.metric("Method PSNR", f"{current_psnr:.2f} dB")
        col4.metric("Method SSIM", f"{current_ssim:.4f}")
        
        st.markdown("---")
        
        st.subheader("Visual Comparison")
        st.markdown(f"Comparing **{selected_method_name.split(' ')[0]}** vs **Native 1080p**")
        
        image_comparison(
            img1=Image.fromarray(upscaled_img),
            img2=Image.fromarray(hr_img),
            label1=selected_method_name.split(" ")[0],
            label2="Native 1080p Ground Truth",
            width=1000,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )
    except Exception as e:
        st.error(f"Error processing {selected_image}. Details: {e}")