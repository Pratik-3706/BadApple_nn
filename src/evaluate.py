"""
Post-training evaluation - checks how well the model actually learned the video.

Run this after training to get per-frame PSNR/SSIM numbers and
side-by-side comparison images. If the numbers look bad,
train longer or bump up the model size.

Usage:
    python src/evaluate.py
"""

import torch
import numpy as np
import cv2
import os
import sys
import time
from skimage.metrics import structural_similarity as ssim


sys.path.insert(0, os.path.dirname(__file__))
from model import build_model
from dataset import make_inference_grid, TARGET_W, TARGET_H


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'badapple_nn.pt')
FRAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'frames.npy')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'train_outputs', 'eval')


def psnr(original, reconstructed):
    """
    Peak Signal-to-Noise Ratio. Higher = better.
    Anything above 25 dB is watchable, 30+ is good, 35+ is great.
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return 50.0  # basically perfect
    return 10.0 * np.log10(1.0 / mse)


def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # load checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"no checkpoint found at {CHECKPOINT_PATH}")
        print("run train.py first!")
        sys.exit(1)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    print(f"loaded checkpoint from epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.6f}")

    total_frames = checkpoint['total_frames']

    # build model + load weights
    model = build_model(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # load original frames for comparison
    original_frames = np.load(FRAMES_PATH)
    print(f"original frames: {original_frames.shape}")

    psnr_values = []
    ssim_values = []
    t0 = time.time()

    print(f"\nreconstructing {total_frames} frames...")

    with torch.no_grad():
        for fi in range(total_frames):
            grid = make_inference_grid(fi, total_frames, device=device)
            logits = model(grid)

            # the model outputs raw logits (trained with BCEWithLogitsLoss).
            # i need to apply the same sigmoid(x * 4.0) that the inference
            # engine uses, otherwise i'm comparing logits against [0,1] pixels
            # and the metrics are garbage.
            pixels = torch.sigmoid(logits * 4.0)

            # reshape to image
            recon = pixels.cpu().numpy().reshape(TARGET_H, TARGET_W)
            original = original_frames[fi]

            frame_psnr = psnr(original, recon)
            frame_ssim = ssim(original, recon, data_range=1.0)
            psnr_values.append(frame_psnr)
            ssim_values.append(frame_ssim)

            # save comparison for a handful of frames
            if fi % 500 == 0 or fi == total_frames - 1:
                # side by side: original | reconstructed
                orig_u8 = (original * 255).clip(0, 255).astype(np.uint8)
                recon_u8 = (recon * 255).clip(0, 255).astype(np.uint8)
                combined = np.hstack([orig_u8, recon_u8])
                path = os.path.join(OUTPUT_DIR, f'compare_frame_{fi:05d}.png')
                cv2.imwrite(path, combined)

            if fi % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  frame {fi:5d}/{total_frames} | PSNR: {frame_psnr:.1f} dB | SSIM: {frame_ssim:.4f} | {elapsed:.1f}s")

    elapsed = time.time() - t0
    psnr_arr = np.array(psnr_values)
    ssim_arr = np.array(ssim_values)

    print(f"\n{'='*50}")
    print(f"evaluation done in {elapsed:.1f}s")
    print(f"  avg PSNR:  {psnr_arr.mean():.2f} dB (min: {psnr_arr.min():.2f}, max: {psnr_arr.max():.2f})")
    print(f"  avg SSIM:  {ssim_arr.mean():.4f} (min: {ssim_arr.min():.4f}, max: {ssim_arr.max():.4f})")
    print(f"  std PSNR:  {psnr_arr.std():.2f} dB")
    print(f"  frames < 20 dB: {np.sum(psnr_arr < 20)}")
    print(f"  frames < 25 dB: {np.sum(psnr_arr < 25)}")
    print(f"  frames > 30 dB: {np.sum(psnr_arr > 30)}")
    print(f"\ncomparison images saved to {OUTPUT_DIR}")

    # save curves as numpy for plotting later if needed
    np.save(os.path.join(OUTPUT_DIR, 'psnr_curve.npy'), psnr_arr)
    np.save(os.path.join(OUTPUT_DIR, 'ssim_curve.npy'), ssim_arr)
    print(f"PSNR curve saved to {os.path.join(OUTPUT_DIR, 'psnr_curve.npy')}")
    print(f"SSIM curve saved to {os.path.join(OUTPUT_DIR, 'ssim_curve.npy')}")


if __name__ == '__main__':
    evaluate()
