import os
import math
import random
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# ============================================================
# CONFIGURATION
# ============================================================
HR_DIR = "HR"
LR_DIR = "LR"
SCALE = 4

EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 2e-4

LR_PATCH_SIZE = 48
HR_PATCH_SIZE = LR_PATCH_SIZE * SCALE

NUM_FEATURES = 64
NUM_BLOCKS = 8
VAL_RATIO = 0.1
NUM_WORKERS = 0
SEED = 42

BEST_MODEL = "minecraft_ai_x4_best.pth"
LATEST_MODEL = "minecraft_ai_x4_latest.pth"

# ============================================================
# SEED & GPU SETUP
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# ============================================================
# LOSS ENGINE: L1 + SOBEL GRADIENT (EDGE) LOSS
# ============================================================
class SobelEdgeLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

        self.kernel_x = sobel_x.repeat(3, 1, 1, 1).to(device)
        self.kernel_y = sobel_y.repeat(3, 1, 1, 1).to(device)

    def forward(self, pred, target):
        grad_pred_x = F.conv2d(pred, self.kernel_x, padding=1, groups=3)
        grad_pred_y = F.conv2d(pred, self.kernel_y, padding=1, groups=3)
        grad_target_x = F.conv2d(target, self.kernel_x, padding=1, groups=3)
        grad_target_y = F.conv2d(target, self.kernel_y, padding=1, groups=3)

        loss_x = F.l1_loss(grad_pred_x, grad_target_x)
        loss_y = F.l1_loss(grad_pred_y, grad_target_y)
        return loss_x + loss_y


class CompositeLoss(nn.Module):
    def __init__(self, device, edge_weight=0.2):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.edge_loss = SobelEdgeLoss(device)
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        pixel_loss = self.l1_loss(pred, target)
        edge_penalty = self.edge_loss(pred, target)
        return pixel_loss + (self.edge_weight * edge_penalty)

# ============================================================
# ARCHITECTURE (MinecraftSR)
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
    """
    FIX: previously this always ran exactly two PixelShuffle(2) stages
    (hardcoded `for _ in range(2)`), which only produced correct output
    for SCALE=4. It now derives the number of x2 stages from `scale`,
    and asserts scale is a power of two (PixelShuffle-based upsampling
    requires this).
    """
    def __init__(self, channels, scale=4):
        super().__init__()
        assert scale > 0 and (scale & (scale - 1)) == 0, \
            f"UpsampleBlock scale must be a power of 2, got {scale}"
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
    """
    FIX: removed the torch.clamp(x, 0, 1) that used to sit inside forward().
    Clamping during training zeroes the gradient for any pixel that
    saturates early, which can permanently stall those pixels. The model
    now returns raw values and is trained with L1/edge loss against
    [0,1]-range targets, which naturally pulls outputs into range.
    Clamp explicitly at inference/export time instead (see `predict`).
    """
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
        """Use this at inference time — clamps to a valid image range."""
        self.eval()
        return torch.clamp(self.forward(x), 0.0, 1.0)

# ============================================================
# DATASET & PATCH PIPELINE
# ============================================================
class MinecraftDataset(Dataset):
    """
    FIX (safety):
      - cv2.imread failures (missing/corrupt files) now raise a clear
        error naming the file, instead of crashing later inside cvtColor.
      - Images smaller than the LR patch size now raise a clear error
        instead of a confusing random.randint ValueError.
      - At init, verifies every HR/LR pair is aligned (HR dims == LR dims
        * scale) using PIL's lazy header read (fast — no full decode),
        so a misaligned pair fails loudly instead of silently teaching
        the model to output blur.
    """
    def __init__(self, hr_dir, lr_dir, patch_size=48, scale=4, training=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.training = training

        hr_files = set(os.listdir(hr_dir))
        lr_files = set(os.listdir(lr_dir))
        self.image_files = sorted([
            f for f in hr_files.intersection(lr_files)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ])

        if len(self.image_files) == 0:
            raise RuntimeError("No matching HR/LR image pairs found!")

        self._validate_alignment()

    def _validate_alignment(self):
        bad_pairs = []
        for f in self.image_files:
            hr_path = os.path.join(self.hr_dir, f)
            lr_path = os.path.join(self.lr_dir, f)
            with Image.open(hr_path) as hr_img, Image.open(lr_path) as lr_img:
                hr_w, hr_h = hr_img.size
                lr_w, lr_h = lr_img.size
            if (hr_w, hr_h) != (lr_w * self.scale, lr_h * self.scale):
                bad_pairs.append((f, (lr_w, lr_h), (hr_w, hr_h)))
            if lr_w < self.patch_size or lr_h < self.patch_size:
                bad_pairs.append((f, (lr_w, lr_h), "smaller than patch size"))

        if bad_pairs:
            msg = "\n".join(f"  {f}: LR={lr}, HR={hr}" for f, lr, hr in bad_pairs[:20])
            raise RuntimeError(
                f"Found {len(bad_pairs)} misaligned or too-small HR/LR pair(s) "
                f"(expected HR == LR * {self.scale}, LR >= {self.patch_size}px):\n{msg}"
                + ("\n  ... (truncated)" if len(bad_pairs) > 20 else "")
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        hr_path = os.path.join(self.hr_dir, filename)
        lr_path = os.path.join(self.lr_dir, filename)

        hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        lr = cv2.imread(lr_path, cv2.IMREAD_COLOR)

        if hr is None:
            raise RuntimeError(f"Failed to read HR image: {hr_path}")
        if lr is None:
            raise RuntimeError(f"Failed to read LR image: {lr_path}")

        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        lr_h, lr_w = lr.shape[:2]

        if self.training:
            ps = self.patch_size
            max_y = lr_h - ps
            max_x = lr_w - ps

            lr_y = random.randint(0, max_y)
            lr_x = random.randint(0, max_x)
            hr_y = lr_y * self.scale
            hr_x = lr_x * self.scale

            lr = lr[lr_y:lr_y + ps, lr_x:lr_x + ps]
            hr = hr[hr_y:hr_y + ps * self.scale, hr_x:hr_x + ps * self.scale]

            if random.random() < 0.5:
                lr, hr = np.ascontiguousarray(lr[:, ::-1]), np.ascontiguousarray(hr[:, ::-1])
            if random.random() < 0.5:
                lr, hr = np.ascontiguousarray(lr[::-1, :]), np.ascontiguousarray(hr[::-1, :])
            if random.random() < 0.25:
                lr, hr = np.ascontiguousarray(np.rot90(lr)), np.ascontiguousarray(np.rot90(hr))

        lr = torch.from_numpy(lr.transpose(2, 0, 1)).float() / 255.0
        hr = torch.from_numpy(hr.transpose(2, 0, 1)).float() / 255.0
        return lr, hr

# ============================================================
# TRAINING SETUP
# ============================================================
# FIX: the original code did
#     train_dataset, val_dataset = random_split(dataset, [...])
#     val_dataset.dataset.training = False
# random_split does NOT clone the underlying dataset — both splits shared
# the same `dataset` object, so setting `.training = False` on the val
# split silently disabled cropping/augmentation for the TRAIN split too,
# which would either crash the collate step on variable-sized images or
# train without any patching/augmentation at all.
#
# Fix: use two separate MinecraftDataset instances (one training=True,
# one training=False) over the same files, and split via shared shuffled
# indices with torch.utils.data.Subset so they stay independent.
train_pool = MinecraftDataset(HR_DIR, LR_DIR, patch_size=LR_PATCH_SIZE, scale=SCALE, training=True)
val_pool = MinecraftDataset(HR_DIR, LR_DIR, patch_size=LR_PATCH_SIZE, scale=SCALE, training=False)

num_samples = len(train_pool)
val_size = max(1, int(num_samples * VAL_RATIO))
train_size = num_samples - val_size

indices = list(range(num_samples))
random.Random(SEED).shuffle(indices)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(train_pool, train_indices)
val_dataset = Subset(val_pool, val_indices)

pin_memory = torch.cuda.is_available()
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0)
)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=pin_memory,
    persistent_workers=(NUM_WORKERS > 0)
)

model = MinecraftSR(num_features=NUM_FEATURES, num_blocks=NUM_BLOCKS, scale=SCALE).to(device)
criterion = CompositeLoss(device=device, edge_weight=0.25)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return 100.0
    return (10.0 * torch.log10(1.0 / mse)).item()

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
best_val_loss = float("inf")
print("\n" + "=" * 60)
print("STARTING TRAINING (WITH SOBEL GRADIENT LOSS)")
print("=" * 60 + "\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for lr_imgs, hr_imgs in train_loader:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0

    with torch.no_grad():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)

            val_loss += loss.item()
            # Clamp only for the PSNR metric (not for the loss/gradient path) —
            # PSNR is only meaningful over the valid [0,1] image range.
            val_psnr += calculate_psnr(outputs.clamp(0.0, 1.0).float(), hr_imgs.float())

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)
    scheduler.step()

    print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val PSNR: {val_psnr:.2f} dB")

    torch.save({"model_state_dict": model.state_dict()}, LATEST_MODEL)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"model_state_dict": model.state_dict()}, BEST_MODEL)
        print("    --> New best model checkpoint saved!")

print(f"\nTraining Complete. Best weights saved to {BEST_MODEL}")