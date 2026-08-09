"""
Dataset and coordinate grid utilities for model training.

The model doesn't see "frames" — it sees coordinates. For any point
(t, x, y) in normalized space [-1, 1]^3, it predicts the brightness.
This module handles the mapping between frame indices / pixel positions
and those normalized coordinates.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os


TARGET_W = 480
TARGET_H = 360


class BadAppleDataset(Dataset):
    """
    Loads the numpy frame array and serves random (t, x, y) -> pixel samples.

    Each __getitem__ returns a single coordinate-pixel pair, but in practice
    we use huge batch sizes (64K+) so the dataloader handles batching.
    """

    def __init__(self, npy_path=None):
        if npy_path is None:
            npy_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'frames.npy')

        print(f"loading frames from {npy_path}...")
        self.frames = np.load(npy_path)  # shape: [num_frames, H, W]
        self.num_frames, self.H, self.W = self.frames.shape
        print(f"loaded {self.num_frames} frames, {self.H}x{self.W}")

        # precompute the coordinate grids — we'll index into these
        # x goes left-right [-1, 1], y goes top-bottom [-1, 1]
        xs = np.linspace(-1, 1, self.W, dtype=np.float32)
        ys = np.linspace(-1, 1, self.H, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)  # both shape [H, W]

        # t values for each frame, normalized to [-1, 1]
        self.t_values = np.linspace(-1, 1, self.num_frames, dtype=np.float32)

        # for motion-aware sampling: compute how much each frame
        # differs from the previous one. frames with big changes
        # get sampled more often during training.
        self._compute_frame_weights()

        # total number of possible samples (every pixel in every frame)
        self.total_pixels = self.num_frames * self.H * self.W

    def _compute_frame_weights(self):
        """
        Frames with lots of motion should get sampled more often,
        otherwise fast-moving scenes come out blurry. Simple approach:
        weight each frame by its absolute difference from neighbors.
        """
        diffs = np.zeros(self.num_frames, dtype=np.float32)
        for i in range(1, self.num_frames):
            diffs[i] = np.mean(np.abs(self.frames[i] - self.frames[i-1]))

        # smooth it out a bit so static frames still get some love
        diffs = diffs + 0.1  # baseline weight so nothing gets zero
        self.frame_weights = diffs / diffs.sum()

    def __len__(self):
        # not the real size — we sample randomly anyway.
        # this just controls how many samples = "one epoch"
        return self.total_pixels

    def get_batch(self, batch_size):
        # pick random frames using the motion-aware weights we computed
        frame_indices = np.random.choice(self.num_frames, size=batch_size, p=self.frame_weights)

        # pick random pixels in those frames
        py = np.random.randint(0, self.H, size=batch_size)
        px = np.random.randint(0, self.W, size=batch_size)

        t = self.t_values[frame_indices]
        x = self.grid_x[py, px]
        y = self.grid_y[py, px]
        pixels = self.frames[frame_indices, py, px]

        coords = np.stack([t, x, y], axis=-1).astype(np.float32)
        targets = pixels[..., None].astype(np.float32)

        return torch.from_numpy(coords), torch.from_numpy(targets)


def make_inference_grid(frame_idx, total_frames, device='cpu'):
    """
    Build the full coordinate grid for a single frame.

    During inference we need ALL pixels for one frame, not random samples.
    Returns a tensor of shape [H*W, 3] where each row is (t, x, y).
    """
    t = -1.0 + 2.0 * frame_idx / max(total_frames - 1, 1)

    xs = np.linspace(-1, 1, TARGET_W, dtype=np.float32)
    ys = np.linspace(-1, 1, TARGET_H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # flatten and stack: each pixel becomes one (t, x, y) triplet
    num_pixels = TARGET_H * TARGET_W
    coords = np.stack([
        np.full(num_pixels, t, dtype=np.float32),
        grid_x.flatten(),
        grid_y.flatten(),
    ], axis=1)

    return torch.from_numpy(coords).to(device)


if __name__ == '__main__':
    # quick test — make sure the dataset loads and shapes are right
    ds = BadAppleDataset()
    coords, target = ds[0]
    print(f"sample: coords={coords.shape}, target={target.shape}")
    print(f"  coords: t={coords[0]:.3f}, x={coords[1]:.3f}, y={coords[2]:.3f}")
    print(f"  pixel: {target[0]:.3f}")

    # test inference grid
    grid = make_inference_grid(0, ds.num_frames)
    print(f"\ninference grid shape: {grid.shape}")
    print(f"  (should be [{TARGET_H * TARGET_W}, 3])")
