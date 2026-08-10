"""
Live inference engine - the heart of this whole project.

Loads the trained model, runs real forward passes frame by frame,
and captures the actual activation values from each layer using
PyTorch forward hooks. Nothing is faked or pre-rendered.

This module is used by server.py but can also run standalone
as a headless benchmark to check fps before wiring up the UI.

Usage (headless benchmark):
    python src/inference_engine.py
"""

import torch
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import SineMLP, SineLayer, build_model
from dataset import make_inference_grid, TARGET_W, TARGET_H


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'badapple_nn.pt')


class InferenceEngine:
    """
    Wraps the trained model for live frame generation.

    Call generate_frame(frame_index) and you get back:
      - pixel data (numpy uint8 array, ready for the canvas)
      - activation dict (real values from every hidden layer)
      - timing info (how long the forward pass took)

    The hooks are registered once at startup and stay attached.
    Each forward pass overwrites the activation dict automatically.
    """

    def __init__(self, checkpoint_path=None, device=None):
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_PATH

        # force cuda - if it fails, we want it to crash with an error, not fall back to slow CPU
        if device is None:
            device = torch.device('cuda')
        self.device = device
        print(f"inference engine using: {device}")

        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.total_frames = checkpoint['total_frames']

        # build model and load weights
        self.model = build_model(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # HUGE performance optimization for the user's GPU:
        # SineMLP weights and bounds are well within float16 limits.
        # This halves the VRAM bandwidth and utilizes Tensor Cores.
        if device.type == 'cuda':
            self.model = self.model.to(torch.float16)
            
        self.model.eval()

        info = self.model.get_architecture_info()
        print(f"model loaded: {info['total_params']:,} params, "
              f"trained for {checkpoint['epoch']} epochs, "
              f"loss {checkpoint['loss']:.6f}")

        # this is where the hook data lands after each forward pass.
        # keys = layer names, values = activation arrays.
        # gets overwritten every frame - that's the whole point.
        self.activations = {}

        # register hooks on all SineLayer modules.
        # these fire automatically during forward() and capture
        # the output tensor of each layer. no extra work needed
        # per frame - it just happens.
        self._register_hooks()

        # pre-build coordinate grids for common frame indices.
        # we reuse the same grid object each frame (only t changes),
        # but make_inference_grid handles that for us.
        self.width = TARGET_W
        self.height = TARGET_H

        print(f"ready! {self.total_frames} frames, {self.width}x{self.height}")

    def _register_hooks(self):
        """
        Attach forward hooks to every SineLayer in the network.

        Each hook captures the output of that layer - the raw activation
        values produced by sin(omega * (Wx + b)). Instead of averaging
        across all pixels (which cancels out the sine waves to near-zero),
        we extract the activations for the exact center pixel of the screen.

        These are REAL activations, but represent exactly ONE pixel's computation.
        """
        self._hooks = []
        layer_idx = 0

        for name, module in self.model.named_modules():
            if isinstance(module, SineLayer):
                # closure trick - need to capture layer_idx by value, not ref
                hook = module.register_forward_hook(self._make_hook(f'layer_{layer_idx}'))
                self._hooks.append(hook)
                layer_idx += 1

        print(f"registered hooks on {layer_idx} layers")

    def _make_hook(self, layer_name):
        """
        Returns a hook function that stores activations under the given name.

        The hooks capture the raw tensor after sin(omega * Wx + b).
        During inference on a frame grid (480x360), this tensor is shape
        [172800, 512]. We extract the exact activations for the center pixel
        (row 180, col 240) to send to the frontend.
        """
        def hook_fn(module, input, output):
            # detach from computation graph
            # True center pixel: row 180, col 240 -> index = 180 * 480 + 240 = 86640
            # (Previously there was an off-by-240 bug picking row 180 col 0)
            center_idx = (self.height // 2) * self.width + (self.width // 2)
            acts = output[center_idx].detach().cpu().numpy()
            self.activations[layer_name] = acts.tolist()
        return hook_fn

    def generate_frame(self, frame_index):
        """
        Run one forward pass for the given frame.

        Returns a dict with everything the frontend needs:
        - pixels: base64-ready uint8 array
        - activations: per-layer neuron activation values
        - inference_ms: how long the forward pass took
        - frame_index, total_frames: for the UI counter
        """
        if frame_index >= self.total_frames:
            # past the end of the video - return a black screen and zeroed activations
            img_uint8 = np.zeros((self.height, self.width), dtype=np.uint8)
            zero_acts = {}
            if not self.activations:
                info = self.model.get_architecture_info()
                for i in range(len(info['layer_names'])):
                    zero_acts[f'layer_{i}'] = [0.0] * info['hidden_features']
            else:
                for k, v in self.activations.items():
                    zero_acts[k] = [0.0] * len(v)

            return {
                'pixels': img_uint8,
                'activations': zero_acts,
                'inference_ms': 0.0,
                'frame_index': self.total_frames - 1,
                'total_frames': self.total_frames,
            }

        # clamp to valid range
        frame_index = max(0, min(frame_index, self.total_frames - 1))

        # HUGE performance optimization: cache the static x, y grid on the GPU
        # and only update the t column in-place. This avoids allocating a 172k-element
        # tensor on the CPU and sending it to the GPU every single frame.
        if getattr(self, '_cached_grid', None) is None:
            from dataset import TARGET_W, TARGET_H
            xs = np.linspace(-1, 1, TARGET_W, dtype=np.float32)
            ys = np.linspace(-1, 1, TARGET_H, dtype=np.float32)
            grid_x, grid_y = np.meshgrid(xs, ys)
            coords = np.stack([
                np.zeros(TARGET_H * TARGET_W, dtype=np.float32), # placeholder for t
                grid_x.flatten(),
                grid_y.flatten(),
            ], axis=1)
            self._cached_grid = torch.from_numpy(coords).to(self.device)
            if self.device.type == 'cuda':
                self._cached_grid = self._cached_grid.to(torch.float16)

        # mathematically calculate t and update the first column of the tensor
        t = -1.0 + 2.0 * frame_index / max(self.total_frames - 1, 1)
        self._cached_grid[:, 0] = t

        grid = self._cached_grid

        t0 = time.perf_counter()

        # THE forward pass - this is where everything happens.
        # hooks fire automatically and populate self.activations.
        with torch.no_grad():
            logits = self.model(grid)
            # We restore sigmoid to fix mid-gray rendering, but multiply logits by 4.0 
            # to maintain the high-contrast sharp edges from the previous raw-logit clip.
            pixels = torch.sigmoid(logits * 4.0)

        # sync GPU if needed (perf_counter won't include async GPU work otherwise)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        inference_ms = (time.perf_counter() - t0) * 1000

        # reshape to image and convert to uint8 directly on the GPU
        # to save massive amounts of CPU time and PCI-e bandwidth
        img_uint8 = (pixels * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy().reshape(self.height, self.width)

        return {
            'pixels': img_uint8,
            'activations': dict(self.activations),  
            'inference_ms': round(inference_ms, 2),
            'frame_index': frame_index,
            'total_frames': self.total_frames,
        }

    def get_model_info(self):
        """Architecture info for the frontend diagram."""
        return self.model.get_architecture_info()

    def get_sparse_weights(self, top_k=2):
        """
        Extracts the strongest real connections from the model weights.
        Returns a dictionary mapping layer names to a list of connections.
        Each layer's list is ordered by destination neuron.
        Format per layer: `[ [src_idx_1, weight_1], [src_idx_2, weight_2], ... ]` for each destination neuron.
        """
        sparse = {}
        layer_idx = 0
        
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    weights = module.weight.cpu().numpy()
                    dest_neurons = weights.shape[0]
                    src_neurons = weights.shape[1]
                    
                    # For the final output layer (px), drawing only 5 lines looks extremely sparse
                    # compared to the 512-wide hidden layers. We'll draw the top 128 connections here.
                    k = min(src_neurons, top_k if dest_neurons > 1 else 128)
                    
                    layer_conns = []
                    for i in range(dest_neurons):
                        w_row = weights[i]
                        # get indices of top K absolute weights
                        top_indices = np.argsort(np.abs(w_row))[-k:]
                        neuron_conns = [[int(idx), float(w_row[idx])] for idx in top_indices]
                        layer_conns.append(neuron_conns)
                        
                    sparse[f'layer_{layer_idx}'] = layer_conns
                    layer_idx += 1
                    
        return sparse

    def cleanup(self):
        """Remove hooks when shutting down."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def benchmark():
    """
    Run a quick benchmark to see if we can hit 30fps.
    No UI, no network - just raw inference speed.
    """
    engine = InferenceEngine()

    print(f"\nbenchmarking 200 frames...")
    times = []

    # warm up first few passes are always slow
    for i in range(5):
        engine.generate_frame(i)

    for i in range(200):
        result = engine.generate_frame(i)
        times.append(result['inference_ms'])

    times = np.array(times)
    print(f"\nresults ({len(times)} frames):")
    print(f"  avg: {times.mean():.2f} ms ({1000/times.mean():.0f} fps)")
    print(f"  min: {times.min():.2f} ms")
    print(f"  max: {times.max():.2f} ms")
    print(f"  p95: {np.percentile(times, 95):.2f} ms")
    print(f"  p99: {np.percentile(times, 99):.2f} ms")

    target_fps = 30
    target_ms = 1000 / target_fps
    pct_under = (times < target_ms).mean() * 100
    print(f"\n  frames under {target_ms:.1f}ms ({target_fps}fps target): {pct_under:.1f}%")

    if pct_under > 95:
        print(f"  -> easily hitting {target_fps} fps")
    elif pct_under > 80:
        print(f"  -> mostly hitting {target_fps} fps, some drops")
    else:
        print(f"  -> struggling with {target_fps} fps, consider lowering resolution")

    # check activation payload size
    result = engine.generate_frame(0)
    total_floats = sum(len(v) for v in result['activations'].values())
    print(f"\nactivation payload: {len(result['activations'])} layers, {total_floats} total values")

    engine.cleanup()


if __name__ == '__main__':
    benchmark()
