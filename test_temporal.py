import os
import cv2
import torch
import numpy as np
from temporal_warp import compute_dense_optical_flow, warp_frame_torch, flow_to_hsv_vis

lr_files = sorted([f for f in os.listdir("LR") if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if len(lr_files) < 2:
    print("Need at least 2 consecutive frames in the LR folder to test temporal flow.")
else:
    # Read frame t-1 and frame t
    prev_img = cv2.imread(os.path.join("LR", lr_files[0]))
    curr_img = cv2.imread(os.path.join("LR", lr_files[1]))
    
    print(f"Analyzing motion between {lr_files[0]} (t-1) and {lr_files[1]} (t)...")
    
    # 1. Compute Flow
    flow = compute_dense_optical_flow(prev_img, curr_img)
    flow_vis = flow_to_hsv_vis(flow)
    cv2.imwrite("motion_vectors_vis.png", cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR))
    
    # 2. Warp frame (t-1) forward to match frame t
    prev_t = torch.from_numpy(prev_img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    flow_t = torch.from_numpy(flow).float().unsqueeze(0)
    
    warped_t = warp_frame_torch(prev_t, flow_t)
    warped_np = (warped_t.squeeze(0).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    cv2.imwrite("warped_prev_frame.png", warped_np)
    
    # 3. Compute Frame Difference to verify alignment
    diff = cv2.absdiff(curr_img, warped_np)
    cv2.imwrite("alignment_error.png", diff)
    
    print("Verification complete! Saved:")
    print(" - motion_vectors_vis.png (direction & magnitude of game movement)")
    print(" - warped_prev_frame.png (t-1 motion-compensated to t)")
    print(" - alignment_error.png (residual delta)")