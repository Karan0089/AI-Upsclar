import streamlit as st
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from streamlit_image_comparison import image_comparison
from PIL import Image

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Super Resolution", layout="wide", initial_sidebar_state="expanded")

# --- 2. Sidebar for Dynamic File Selection ---
st.sidebar.title("Configuration")
st.sidebar.markdown("Select a test frame from your dataset.")

# Automatically scan the HR directory for images
try:
    available_images = sorted([f for f in os.listdir("HR") if f.endswith(('.png', '.jpg', '.jpeg'))])
except FileNotFoundError:
    available_images = []

if not available_images:
    st.sidebar.error("No images found! Make sure 'HR' and 'LR' folders exist in this directory.")
    selected_image = None
else:
    selected_image = st.sidebar.selectbox("Test Sequence Image", available_images)

# --- 3. Main UI & App Logic ---
st.title("Spatio-Temporal Super Resolution Engine")
st.markdown("Evaluating baseline Bicubic upscaling against native High-Resolution textures.")

# Cache the processing so swapping images is instantly responsive
@st.cache_data
def load_and_process_images(image_name):
    hr_path = os.path.join("HR", image_name)
    lr_path = os.path.join("LR", image_name)
    
    # Load and convert BGR (OpenCV default) to RGB (Web standard)
    hr_img = cv2.cvtColor(cv2.imread(hr_path), cv2.COLOR_BGR2RGB)
    lr_img = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
    
    # Generate Bicubic Baseline (Math-based upscaling)
    height, width = hr_img.shape[:2]
    bicubic_img = cv2.resize(lr_img, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Calculate performance metrics
    psnr_val = compute_psnr(hr_img, bicubic_img)
    ssim_val = compute_ssim(hr_img, bicubic_img, channel_axis=2)
    
    return hr_img, lr_img, bicubic_img, psnr_val, ssim_val

# --- 4. Render Dashboard ---
if selected_image:
    try:
        hr_img, lr_img, bicubic_img, bicubic_psnr, bicubic_ssim = load_and_process_images(selected_image)
        
        st.subheader("Quantitative Evaluation")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Input (LR)", f"{lr_img.shape[1]}x{lr_img.shape[0]}")
        col2.metric("Target (HR)", f"{hr_img.shape[1]}x{hr_img.shape[0]}")
        col3.metric("Bicubic PSNR", f"{bicubic_psnr:.2f} dB")
        col4.metric("Bicubic SSIM", f"{bicubic_ssim:.4f}")
        
        st.markdown("---")
        
        st.subheader("Visual Comparison")
        st.markdown(f"**Currently analyzing:** `{selected_image}`")
        
        image_comparison(
            img1=Image.fromarray(bicubic_img),
            img2=Image.fromarray(hr_img),
            label1="Bicubic Upscale (Blurry Baseline)",
            label2="Native 1080p (Target Ground Truth)",
            width=1000,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )
    except Exception as e:
        st.error(f"Error processing {selected_image}. Details: {e}")