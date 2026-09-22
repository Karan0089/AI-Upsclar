import os
import re
import math
import random
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from temporal_warp import compute_dense_optical_flow, warp_frame_torch

# ============================================================
# CONFIGURATION
# ============================================================
HR_DIR = "HR_seq"
LR_DIR = "LR_seq"
SCALE = 4

EPOCHS = 80
BATCH_SIZE = 8          # NOTE: temporal consistency loss runs 2 forward passes
                         # per step (prev + curr), so peak memory is higher than
                         # single-pass pairwise training. Lower this if you OOM.
LEARNING_RATE = 2e-4

LR_PATCH_SIZE = 48
HR_PATCH_SIZE = LR_PATCH_SIZE * SCALE

NUM_FEATURES = 64
NUM_BLOCKS = 8
VAL_RATIO = 0.1
NUM_WORKERS = 0          # Kept at 0 for Windows safety
SEED = 42

# Distinct filenames from other temporal variants you may be comparing against,
# so checkpoints don't silently overwrite each other.
BEST_MODEL = "minecraft_temporal_seq_x4_best.pth"
LATEST_MODEL = "minecraft_temporal_seq_x4_latest.pth"

# ---- temporal pairing config ----
# Adjust this regex to match your actual filenames, e.g. "forest_0001.png" ->
# clip="forest", frame=1. This stops frames from different clips/sessions
# from being paired as if they were consecutive just because they're
# adjacent alphabetically. Files that don't match are treated as isolated
# (never paired) rather than silently mis-paired.
FRAME_ID_PATTERN = re.compile(r"^(?:(?P<clip>.+?)_)?(?P<frame>\d+)\.[^.]+$")
MAX_FRAME_GAP = 1        # only pair frames whose numbers differ by exactly this much

# ---- temporal consistency config ----
FLOW_CACHE_DIR = "flow_cache"           # set to None to disable disk caching of optical flow
USE_TEMPORAL_CONSISTENCY_LOSS = False   # set False to fall back to single-pass (no temporal loss) for debugging
TEMPORAL_LOSS_WEIGHT = 0.5              # starting point — tune based on flicker vs. sharpness trade-off

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
# LOSS ENGINE: L1 + SOBEL GRADIENT LOSS
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
        return F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)


class CompositeLoss(nn.Module):
    def __init__(self, device, edge_weight=0.20):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.edge_loss = SobelEdgeLoss(device)
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        return self.l1_loss(pred, target) + (self.edge_weight * self.edge_loss(pred, target))

# ============================================================
# SPATIO-TEMPORAL ARCHITECTURE (6-Channel Input) — unchanged
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


class MinecraftTemporalSR(nn.Module):
    def __init__(self, in_channels=6, num_features=64, num_blocks=8, scale=4):
        super().__init__()
        # in_channels=6: 3 channels (frame t) + 3 channels (warped frame t-1)
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
        return self.tail(x)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return torch.clamp(self.forward(x), 0.0, 1.0)

# ============================================================
# FLOW UPSAMPLING (needed for the temporal consistency loss)
# ============================================================
def upsample_flow(flow, scale):
    """
    flow: (N, H, W, 2) pixel-unit displacements at LR resolution, in the
    same (dx, dy) convention warp_frame_torch expects.
    Returns: (N, H*scale, W*scale, 2) with displacement magnitudes scaled
    up to match the new resolution.

    ASSUMPTION: matches the (N, H, W, 2) / pixel-unit convention your
    dataset already used with compute_dense_optical_flow + warp_frame_torch.
    Verify against temporal_warp.py if results look off.
    """
    flow_chw = flow.permute(0, 3, 1, 2)  # (N, 2, H, W)
    flow_up = F.interpolate(flow_chw, scale_factor=scale, mode="bilinear", align_corners=False)
    flow_up = flow_up * scale
    return flow_up.permute(0, 2, 3, 1)  # (N, H*scale, W*scale, 2)

# ============================================================
# TEMPORAL DATASET (Clip-aware triplets: t-2, t-1, t)
# ============================================================
class MinecraftTemporalDataset(Dataset):
    """
    Returns triplets (f0, f1, f2) of consecutive frames *within the same
    clip*, so optical flow is never computed across an unrelated scene cut.

    Fixes applied here (reapplied from the previous pass, since this file
    had reverted to plain sorted-order pairwise pairing with no safety
    checks or temporal loss):
      - Frames grouped by clip (parsed via FRAME_ID_PATTERN), only paired
        within a clip, with a max allowed frame-number gap (MAX_FRAME_GAP).
      - cv2.imread failures raise a clear error naming the file.
      - HR/LR alignment + minimum patch size validated at init (PIL lazy
        read) instead of failing silently later.
      - Optical flow cached to disk (FLOW_CACHE_DIR) so it's computed once
        total instead of once per sample per epoch.
      - Flip augmentation flips the flow's sign/axis consistently with the
        image (needed for the temporal loss to warp correctly). 90-degree
        rotation augmentation is left out since rotating flow correctly
        requires swapping x/y components too.
    """
    def __init__(self, hr_dir, lr_dir, patch_size=48, scale=4, training=True,
                 flow_cache_dir=FLOW_CACHE_DIR):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.training = training
        self.flow_cache_dir = flow_cache_dir
        if self.flow_cache_dir is not None:
            os.makedirs(self.flow_cache_dir, exist_ok=True)

        hr_files = set(os.listdir(hr_dir))
        lr_files = set(os.listdir(lr_dir))
        all_frames = sorted([
            f for f in hr_files.intersection(lr_files)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ])

        if len(all_frames) < 3:
            raise RuntimeError(f"Need at least 3 consecutive frames (in one clip) in {hr_dir}/{lr_dir}!")

        self._validate_alignment(all_frames)

        frames_by_clip = defaultdict(list)
        unmatched = 0
        for f in all_frames:
            clip, frame_num = self._parse_clip_and_frame(f)
            if clip is None:
                unmatched += 1
                clip = f"_unmatched_{f}"  # isolated clip of its own -> never paired
                frame_num = 0
            frames_by_clip[clip].append((frame_num, f))

        if unmatched > 0:
            print(f"[MinecraftTemporalDataset] WARNING: {unmatched} file(s) didn't match "
                  f"FRAME_ID_PATTERN and were treated as isolated (unpaired) frames. "
                  f"Update FRAME_ID_PATTERN if this is unexpected.")

        self.triplets = []
        for clip, frame_list in frames_by_clip.items():
            frame_list.sort(key=lambda x: x[0])
            for i in range(2, len(frame_list)):
                n0, f0 = frame_list[i - 2]
                n1, f1 = frame_list[i - 1]
                n2, f2 = frame_list[i]
                if (n1 - n0) <= MAX_FRAME_GAP and (n2 - n1) <= MAX_FRAME_GAP:
                    self.triplets.append((f0, f1, f2))

        if len(self.triplets) == 0:
            raise RuntimeError(
                "No valid 3-frame temporal triplets found. Check that FRAME_ID_PATTERN "
                "matches your filenames and that MAX_FRAME_GAP is appropriate."
            )

    @staticmethod
    def _parse_clip_and_frame(filename):
        m = FRAME_ID_PATTERN.match(filename)
        if m is None:
            return None, None
        clip = m.group("clip") or "default"
        return clip, int(m.group("frame"))

    def _validate_alignment(self, filenames):
        bad = []
        for f in filenames:
            hr_path = os.path.join(self.hr_dir, f)
            lr_path = os.path.join(self.lr_dir, f)
            with Image.open(hr_path) as hr_img, Image.open(lr_path) as lr_img:
                hr_w, hr_h = hr_img.size
                lr_w, lr_h = lr_img.size
            if (hr_w, hr_h) != (lr_w * self.scale, lr_h * self.scale):
                bad.append((f, (lr_w, lr_h), (hr_w, hr_h)))
            elif lr_w < self.patch_size or lr_h < self.patch_size:
                bad.append((f, (lr_w, lr_h), "smaller than patch size"))
        if bad:
            msg = "\n".join(f"  {f}: LR={lr}, HR={hr}" for f, lr, hr in bad[:20])
            raise RuntimeError(
                f"Found {len(bad)} misaligned or too-small HR/LR file(s):\n{msg}"
                + ("\n  ... (truncated)" if len(bad) > 20 else "")
            )

    def __len__(self):
        return len(self.triplets)

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def _get_flow(self, name_prev, name_curr, img_prev, img_curr):
        if self.flow_cache_dir is not None:
            cache_key = f"{name_prev}__{name_curr}".replace(os.sep, "_")
            cache_path = os.path.join(self.flow_cache_dir, cache_key + ".npy")
            if os.path.exists(cache_path):
                return np.load(cache_path)
            flow = compute_dense_optical_flow(img_prev, img_curr)
            np.save(cache_path, flow)
            return flow
        return compute_dense_optical_flow(img_prev, img_curr)

    @staticmethod
    def _warp(img_uint8, flow):
        img_t = torch.from_numpy(img_uint8.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        flow_t = torch.from_numpy(flow).float().unsqueeze(0)
        warped_t = warp_frame_torch(img_t, flow_t).squeeze(0)
        warped = (warped_t.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        return warped

    @staticmethod
    def _to_tensor(img_rgb_uint8):
        return torch.from_numpy(img_rgb_uint8.transpose(2, 0, 1)).float() / 255.0

    def __getitem__(self, index):
        f0, f1, f2 = self.triplets[index]

        lr0 = self._read_image(os.path.join(self.lr_dir, f0))
        lr1 = self._read_image(os.path.join(self.lr_dir, f1))
        lr2 = self._read_image(os.path.join(self.lr_dir, f2))
        hr2 = self._read_image(os.path.join(self.hr_dir, f2))

        flow01 = self._get_flow(f0, f1, lr0, lr1)  # t-2 -> t-1
        flow12 = self._get_flow(f1, f2, lr1, lr2)  # t-1 -> t   (kept for the temporal loss)

        warped0to1 = self._warp(lr0, flow01)
        warped1to2 = self._warp(lr1, flow12)

        lr1_rgb = cv2.cvtColor(lr1, cv2.COLOR_BGR2RGB)
        lr2_rgb = cv2.cvtColor(lr2, cv2.COLOR_BGR2RGB)
        warped0to1_rgb = cv2.cvtColor(warped0to1, cv2.COLOR_BGR2RGB)
        warped1to2_rgb = cv2.cvtColor(warped1to2, cv2.COLOR_BGR2RGB)
        hr2_rgb = cv2.cvtColor(hr2, cv2.COLOR_BGR2RGB)
        flow12 = flow12.astype(np.float32)

        lr_h, lr_w = lr2_rgb.shape[:2]

        if self.training:
            ps = self.patch_size
            lr_y = random.randint(0, lr_h - ps)
            lr_x = random.randint(0, lr_w - ps)
            hr_y, hr_x = lr_y * self.scale, lr_x * self.scale

            lr1_rgb = lr1_rgb[lr_y:lr_y + ps, lr_x:lr_x + ps]
            lr2_rgb = lr2_rgb[lr_y:lr_y + ps, lr_x:lr_x + ps]
            warped0to1_rgb = warped0to1_rgb[lr_y:lr_y + ps, lr_x:lr_x + ps]
            warped1to2_rgb = warped1to2_rgb[lr_y:lr_y + ps, lr_x:lr_x + ps]
            flow12 = flow12[lr_y:lr_y + ps, lr_x:lr_x + ps, :]
            hr2_rgb = hr2_rgb[hr_y:hr_y + ps * self.scale, hr_x:hr_x + ps * self.scale]

            # Synchronized augmentation — the raw flow must flip sign/axis
            # along with the image, or the temporal warp direction breaks.
            if random.random() < 0.5:  # horizontal flip
                lr1_rgb = np.ascontiguousarray(lr1_rgb[:, ::-1])
                lr2_rgb = np.ascontiguousarray(lr2_rgb[:, ::-1])
                warped0to1_rgb = np.ascontiguousarray(warped0to1_rgb[:, ::-1])
                warped1to2_rgb = np.ascontiguousarray(warped1to2_rgb[:, ::-1])
                hr2_rgb = np.ascontiguousarray(hr2_rgb[:, ::-1])
                flow12 = np.ascontiguousarray(flow12[:, ::-1, :])
                flow12[..., 0] *= -1.0
            if random.random() < 0.5:  # vertical flip
                lr1_rgb = np.ascontiguousarray(lr1_rgb[::-1, :])
                lr2_rgb = np.ascontiguousarray(lr2_rgb[::-1, :])
                warped0to1_rgb = np.ascontiguousarray(warped0to1_rgb[::-1, :])
                warped1to2_rgb = np.ascontiguousarray(warped1to2_rgb[::-1, :])
                hr2_rgb = np.ascontiguousarray(hr2_rgb[::-1, :])
                flow12 = np.ascontiguousarray(flow12[::-1, :, :])
                flow12[..., 1] *= -1.0

        input_prev_step = torch.cat([self._to_tensor(lr1_rgb), self._to_tensor(warped0to1_rgb)], dim=0)  # 6ch @ t-1
        input_curr_step = torch.cat([self._to_tensor(lr2_rgb), self._to_tensor(warped1to2_rgb)], dim=0)  # 6ch @ t
        hr2_t = self._to_tensor(hr2_rgb)
        flow12_t = torch.from_numpy(np.ascontiguousarray(flow12)).float()  # (H, W, 2), LR resolution

        return input_prev_step, input_curr_step, flow12_t, hr2_t

# ============================================================
# ENTRY POINT & TRAINING LOOP
# ============================================================
if __name__ == "__main__":
    train_pool = MinecraftTemporalDataset(HR_DIR, LR_DIR, patch_size=LR_PATCH_SIZE, scale=SCALE, training=True)
    val_pool = MinecraftTemporalDataset(HR_DIR, LR_DIR, patch_size=LR_PATCH_SIZE, scale=SCALE, training=False)

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

    model = MinecraftTemporalSR(in_channels=6, num_features=NUM_FEATURES, num_blocks=NUM_BLOCKS, scale=SCALE).to(device)
    criterion = CompositeLoss(device=device, edge_weight=0.20)
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
    print("\n" + "=" * 60)
    print(f"STARTING SPATIO-TEMPORAL TRAINING (6-CH MULTI-FRAME FUSION, temporal_loss={USE_TEMPORAL_CONSISTENCY_LOSS})")
    print("=" * 60 + "\n")

    for epoch in range(EPOCHS):
        model.train()
        train_recon_loss = 0.0
        train_temporal_loss = 0.0

        for input_prev, input_curr, flow12, hr_imgs in train_loader:
            input_prev = input_prev.to(device, non_blocking=True)
            input_curr = input_curr.to(device, non_blocking=True)
            flow12 = flow12.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs_curr = model(input_curr)
                recon_loss = criterion(outputs_curr, hr_imgs)

                if USE_TEMPORAL_CONSISTENCY_LOSS:
                    outputs_prev = model(input_prev)
                    flow_hr = upsample_flow(flow12, SCALE)
                    # Detach the previous step's prediction so this term only
                    # pulls the CURRENT step's output toward consistency with
                    # an already-fixed previous prediction, rather than both
                    # predictions chasing each other.
                    warped_prev_output = warp_frame_torch(outputs_prev.detach(), flow_hr)
                    temporal_loss = F.l1_loss(warped_prev_output, outputs_curr)
                    loss = recon_loss + TEMPORAL_LOSS_WEIGHT * temporal_loss
                else:
                    temporal_loss = torch.tensor(0.0, device=device)
                    loss = recon_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_recon_loss += recon_loss.item()
            train_temporal_loss += temporal_loss.item()

        train_recon_loss /= len(train_loader)
        train_temporal_loss /= len(train_loader)

        # Validation — reconstruction-only, so it's comparable across runs
        # regardless of whether the temporal loss is enabled for that run.
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0

        with torch.no_grad():
            for _, input_curr, _, hr_imgs in val_loader:
                input_curr = input_curr.to(device, non_blocking=True)
                hr_imgs = hr_imgs.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = model(input_curr)
                    loss = criterion(outputs, hr_imgs)

                val_loss += loss.item()
                val_psnr += calculate_psnr(outputs.clamp(0.0, 1.0).float(), hr_imgs.float())

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        scheduler.step()

        print(f"Epoch [{epoch + 1:03d}/{EPOCHS}] | Train Recon: {train_recon_loss:.5f} | "
              f"Train Temporal: {train_temporal_loss:.5f} | Val Loss: {val_loss:.5f} | Val PSNR: {val_psnr:.2f} dB")

        torch.save({"model_state_dict": model.state_dict()}, LATEST_MODEL)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, BEST_MODEL)
            print("    --> New best model checkpoint saved!")

    print(f"\nTraining Complete. Best weights saved to {BEST_MODEL}")