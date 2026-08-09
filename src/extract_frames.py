"""
Extract frames from Bad Apple video and save as numpy array.

This is a run-once script — you do this before training, then never again
unless you change the source video or resolution.
"""

import cv2
import numpy as np
import os
import sys
import time


# ============================================================
# config — change these if you want a different resolution.
# the model doesn't care, it just sees (t, x, y) -> pixel.
# you'll need to retrain after changing resolution though.
# ============================================================
VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', 'bad_apple_vid', 'vid.mp4')
FRAMES_DIR = os.path.join(os.path.dirname(__file__), '..', 'bad_apple_vid', 'extracted_frames')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NPY_PATH = os.path.join(DATA_DIR, 'frames.npy')


TARGET_W = 480
TARGET_H = 360


def extract():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"couldn't open video at {VIDEO_PATH}")
        print("make sure vid.mp4 is in bad_apple_vid/")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {total} frames @ {fps:.1f} fps ({total/fps:.1f}s)")
    print(f"target resolution: {TARGET_W}x{TARGET_H}")

    frames = []
    saved = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # grayscale + resize — order matters, grayscale first is faster
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        # normalize to [0, 1] for training
        normalized = small.astype(np.float32) / 255.0
        frames.append(normalized)

        # save every 100th frame as PNG so we can eyeball the quality
        if saved % 100 == 0:
            png_path = os.path.join(FRAMES_DIR, f'frame_{saved:05d}.png')
            cv2.imwrite(png_path, small)

        saved += 1

        # animated progress bar
        if saved % 10 == 0 or saved == total:
            pct = saved / total
            bar_len = 20
            filled = int(bar_len * pct)
            bar = '#' * filled + ' ' * (bar_len - filled)
            spinner = ['/', '-', '\\', '|'][(saved // 10) % 4]
            print(f"\rextracting [{bar}] {pct*100:5.1f}%   {spinner} ", end="", flush=True)

    cap.release()

    # stack into one big array: shape [num_frames, H, W]
    all_frames = np.stack(frames, axis=0)
    np.save(NPY_PATH, all_frames)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(NPY_PATH) / (1024 * 1024)
    print(f"\ndone! {saved} frames extracted in {elapsed:.1f}s")
    print(f"saved to {NPY_PATH} ({size_mb:.1f} MB)")
    print(f"shape: {all_frames.shape} (frames, H, W)")
    print(f"sample PNGs in {FRAMES_DIR}")


if __name__ == '__main__':
    extract()
