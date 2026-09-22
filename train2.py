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
from torchvision.models import vgg19, VGG19_Weights

# ============================================================
# CONFIGURATION
# ============================================================
HR_DIR = "HR"
LR_DIR = "LR"
SCALE = 4

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 2e-4

LR_PATCH_SIZE = 48
HR_PATCH_SIZE = LR_PATCH_SIZE * SCALE

NUM_FEATURES = 64
NUM_BLOCKS = 16          # Deeper architecture (doubled from 8 to 16)
RES_SCALE = 0.1          # Residual scaling factor to stabilize deep networks
VAL_RATIO = 0.1
NUM_WORKERS = 0          # Windows stability
SEED = 42

BEST_MODEL = "minecraft_spatial_pro_best.pth"
LATEST_MODEL = "minecraft_spatial_pro_latest.pth"

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
# ADVANCED LOSS PIPELINE (L1 + VGG PERCEPTUAL + SOBEL EDGE)
# ============================================================
class VGGPerceptualLoss(nn.Module):
    """
    Extracts high-frequency visual features before pooling layers.
    Forces sharp edges and structural contrast instead of blurry averages.
    """
    def __init__(self, device):
        super().__init__()
        weights = VGG19_Weights.DEFAULT
        vgg = vgg19(weights=weights).features[:16].to(device) # Up to relu3_4
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        
        # ImageNet normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        return F.l1_loss(pred_features, target_features)

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

        return F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

class ComprehensiveLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.vgg = VGGPerceptualLoss(device)
        self.sobel = SobelEdgeLoss(device)

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        vgg_loss = self.vgg(pred, target)
        sobel_loss = self.sobel(pred, target)
        # Weighted combination to retain color accuracy (L1) while enforcing textures (VGG + Sobel)
        return l1_loss + (0.10 * vgg_loss) + (0.15 * sobel_loss)

# ============================================================
# DEEP RESIDUAL ARCHITECTURE (EDSR-STYLE)
# ============================================================
class ResidualBlockScaled(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + (self.block(x) * self.res_scale)

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

class MinecraftSpatialPro(nn.Module):
    def __init__(self, num_features=64, num_blocks=16, scale=4, res_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlockScaled(num_features, res_scale) for _ in range(num_blocks)])
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
        return self.tail(x)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return torch.clamp(self.forward(x), 0.0, 1.0)

# ============================================================
# DATASET PIPELINE
# ============================================================
class MinecraftDataset(Dataset):
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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        hr_path = os.path.join(self.hr_dir, filename)
        lr_path = os.path.join(self.lr_dir, filename)

        hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        lr = cv2.imread(lr_path, cv2.IMREAD_COLOR)

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

            # Augmentations
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
# MAIN TRAINING EXECUTION
# ============================================================
if __name__ == "__main__":
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
        num_workers=NUM_WORKERS, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory
    )

    model = MinecraftSpatialPro(
        num_features=NUM_FEATURES, num_blocks=NUM_BLOCKS, scale=SCALE, res_scale=RES_SCALE
    ).to(device)
    
    criterion = ComprehensiveLoss(device=device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def calculate_psnr(pred, target):
        mse = torch.mean((pred - target) ** 2)
        if mse.item() == 0:
            return 100.0
        return (10.0 * torch.log10(1.0 / mse)).item()

    best_val_loss = float("inf")
    print("\n" + "=" * 65)
    print("STARTING ADVANCED SPATIO TRAINING (16 BLOCKS + VGG PERCEPTUAL)")
    print("=" * 65 + "\n")

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
                val_psnr += calculate_psnr(outputs.clamp(0.0, 1.0).float(), hr_imgs.float())

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        scheduler.step()

        print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val PSNR: {val_psnr:.2f} dB")

        torch.save({"model_state_dict": model.state_dict()}, LATEST_MODEL)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, BEST_MODEL)
            print("    --> New BEST model checkpoint saved!")

    print(f"\nTraining Complete. Best weights saved to {BEST_MODEL}")