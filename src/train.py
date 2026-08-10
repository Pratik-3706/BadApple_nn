"""
Training loop for the sinusoidal MLP.

This runs offline, once, before you can do anything live.
Expect 1-3 hours on an RTX 2050 depending on how picky you are
about quality. The model memorizes the entire video frame by frame.

Usage:
    python src/train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import cv2

# local imports
sys.path.insert(0, os.path.dirname(__file__))
from model import build_model
from dataset import BadAppleDataset, make_inference_grid, TARGET_W, TARGET_H


# ============================================================
# training config
# ============================================================
BATCH_SIZE = 65536          # 64K samples per step - covers ~1.5 full frames worth of pixels
LEARNING_RATE = 1e-4
EPOCHS = 600                # usually converges well before this
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'train_outputs')

# how often to save sample frames during training (in epochs)
SAMPLE_EVERY = 10
# which frames to reconstruct as samples (spread across the video)
SAMPLE_FRAME_INDICES = [0, 500, 1500, 3000, 5000]


def save_sample_frames(model, total_frames, epoch, device):
    """
    Reconstruct a few specific frames and save them as PNGs.
    This is how you tell if training is actually working without
    waiting for it to finish. Check train_outputs/ periodically.
    """
    model.eval()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with torch.no_grad():
        for fi in SAMPLE_FRAME_INDICES:
            if fi >= total_frames:
                continue

            grid = make_inference_grid(fi, total_frames, device=device)
            pixels = torch.sigmoid(model(grid) * 4.0)

            # reshape back to image
            img = pixels.cpu().numpy().reshape(TARGET_H, TARGET_W)
            img = (img * 255).clip(0, 255).astype(np.uint8)

            path = os.path.join(OUTPUT_DIR, f'epoch{epoch:04d}_frame{fi:05d}.png')
            cv2.imwrite(path, img)

    model.train()


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # pick device - cuda if available, otherwise cpu (slower but works)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # build model
    model = build_model(device)
    info = model.get_architecture_info()
    print(f"model: {info['hidden_layers']} layers x {info['hidden_features']} wide = {info['total_params']:,} params")

    # load data
    dataset = BadAppleDataset()
    total_frames = dataset.num_frames

    # We bypass DataLoader entirely because it introduces massive python 
    # overhead for coordinate-based dataset sampling.
    # We cap steps_per_epoch to 2000 so the progress bar moves quickly and 
    # checkpoints save every few minutes, instead of waiting 30 mins per epoch.
    steps_per_epoch = min(len(dataset) // BATCH_SIZE, 2000)

    # optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'badapple_nn.pt')
    latest_path = os.path.join(CHECKPOINT_DIR, 'badapple_nn_latest.pt')

    start_epoch = 1

    # prefer latest checkpoint (most recent epoch) over best checkpoint.
    # this way we never repeat work after a restart.
    resume_path = None
    if os.path.exists(latest_path):
        resume_path = latest_path
    elif os.path.exists(checkpoint_path):
        resume_path = checkpoint_path

    if resume_path:
        print(f"found checkpoint at {resume_path}, attempting to resume...")
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_loss = ckpt.get('loss', float('inf'))
            print(f"successfully resumed from epoch {start_epoch - 1} with loss {best_loss:.6f}")

            # if we resumed from latest, we need the best loss from the best checkpoint
            # (since latest tracks its own loss, not the all-time best)
            if resume_path == latest_path and os.path.exists(checkpoint_path):
                best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
                best_loss = best_ckpt.get('loss', float('inf'))
                print(f"best loss from best checkpoint: {best_loss:.6f}")
        except Exception as e:
            print(f"failed to load checkpoint: {e}. starting fresh.")
            start_epoch = 1

    # enable TF32 for a huge speedup on Ampere GPUs 
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\nstarting training: {EPOCHS} epochs, {BATCH_SIZE} batch size")
    print(f"total pixels in dataset: {len(dataset):,}")
    print(f"steps per epoch: {steps_per_epoch}")
    print("-" * 60)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for _ in range(steps_per_epoch):
            # active learning logic: sample way more pixels than needed.
            # find the ones the model gets wrong, and only train on those.
            oversample = 4
            coords_big, targets_big = dataset.get_batch(BATCH_SIZE * oversample)
            coords_big = coords_big.to(device)
            targets_big = targets_big.to(device)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    pred_big = model(coords_big)
                    # compute unreduced loss to find exact error per pixel
                    raw_loss = nn.functional.binary_cross_entropy_with_logits(
                        pred_big, targets_big, reduction='none'
                    ).squeeze(-1)
            
            # hard negative mining. take the hardest 90 percent.
            hard_count = int(BATCH_SIZE * 0.9)
            rand_count = BATCH_SIZE - hard_count
            
            _, hard_idx = torch.topk(raw_loss, hard_count)
            
            # fill the rest with random pixels to prevent catastrophic forgetting of the background.
            rand_idx = torch.randint(0, BATCH_SIZE * oversample, (rand_count,), device=device)
            
            final_idx = torch.cat([hard_idx, rand_idx])
            
            coords = coords_big[final_idx]
            targets = targets_big[final_idx]

            optimizer.zero_grad(set_to_none=True)
            
            # Automatic Mixed Precision (AMP) cuts memory usage in half and 
            # uses the GPU's Tensor Cores to nearly double training speed.
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred = model(coords)
                # Model guesses wrong: function angry, function beats model.
                # Model guesses right: function happy, function rewards model.
                loss = loss_fn(pred, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0 or num_batches == steps_per_epoch:
                pct = num_batches / steps_per_epoch
                bar_len = 20
                filled = int(bar_len * pct)
                bar = '#' * filled + ' ' * (bar_len - filled)
                print(f"\repoch {epoch:3d}/{EPOCHS} [{bar}] {pct*100:5.1f}% | loss: {loss.item():.4f} ", end="", flush=True)

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'total_frames': total_frames,
                'target_w': TARGET_W,
                'target_h': TARGET_H,
            }, checkpoint_path)
            marker = ' * saved best'
        else:
            marker = ''

        # always save latest checkpoint so we never lose progress
        latest_path = os.path.join(CHECKPOINT_DIR, 'badapple_nn_latest.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'total_frames': total_frames,
            'target_w': TARGET_W,
            'target_h': TARGET_H,
        }, latest_path)

        # \r to overwrite the progress bar cleanly
        print(f"\repoch {epoch:4d}/{EPOCHS} | loss: {avg_loss:.6f} | "
              f"lr: {lr:.2e} | {elapsed:.1f}s{marker}" + " " * 15)

        # dump sample frames so I can see progress
        if epoch % SAMPLE_EVERY == 0 or epoch == 1:
            save_sample_frames(model, total_frames, epoch, device)
            print(f"  -> sample frames saved to {OUTPUT_DIR}")

    print(f"\ndone! best loss: {best_loss:.6f}")
    print(f"checkpoint: {checkpoint_path}")
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"checkpoint size: {size_mb:.1f} MB")


if __name__ == '__main__':
    train()
