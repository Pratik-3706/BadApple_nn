"""
Live inference engine — the heart of this whole project.

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

        # force cuda — if it fails, we want it to crash with an error, not fall back to slow CPU
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
        self.model.eval()

        info = self.model.get_architecture_info()
        print(f"model loaded: {info['total_params']:,} params, "
              f"trained for {checkpoint['epoch']} epochs, "
              f"loss {checkpoint['loss']:.6f}")

        # this is where the hook data lands after each forward pass.
        # keys = layer names, values = activation arrays.
        # gets overwritten every frame — that's the whole point.
        self.activations = {}

        # register hooks on all SineLayer modules.
        # these fire automatically during forward() and capture
        # the output tensor of each layer. no extra work needed
        # per frame — it just happens.
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

        Each hook captures the output of that layer — the raw activation
        values produced by sin(omega * (Wx + b)). We reduce them to
        per-neuron means (average across all spatial positions) to keep
        the payload small enough to stream at 30fps.

        These are the REAL activations, not simulated. The exact numbers
        that produced the frame you're looking at.
        """
        self._hooks = []
        layer_idx = 0

        for name, module in self.model.named_modules():
            if isinstance(module, SineLayer):
                # closure trick — need to capture layer_idx by value, not ref
                hook = module.register_forward_hook(self._make_hook(f'layer_{layer_idx}'))
                self._hooks.append(hook)
                layer_idx += 1

        print(f"registered hooks on {layer_idx} layers")

    def _make_hook(self, layer_name):
        """
        Returns a hook function that stores activations under the given name.

        The hook receives (module, input, output) from PyTorch.
        output shape is [num_pixels, hidden_dim] — for 240x180 that's
        [43200, 256]. We take the mean across pixels to get one value
        per neuron: [256]. That's what goes to the frontend.
        """
        def hook_fn(module, input, output):
            # detach from computation graph
            # Instead of averaging across all 172,800 pixels (which makes the animation
            # look static/zeroed-out), we take the mathematically exact activation
            # of the neuron for the exact center pixel of the screen.
            center_idx = output.shape[0] // 2
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
            # past the end of the video — return a black screen and zeroed activations
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

        # build the coordinate grid for this frame
        grid = make_inference_grid(frame_index, self.total_frames, device=self.device)

        t0 = time.perf_counter()

        # THE forward pass — this is where everything happens.
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

        # reshape to image and convert to uint8
        img = pixels.cpu().numpy().reshape(self.height, self.width)
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

        return {
            'pixels': img_uint8,
            'activations': dict(self.activations),  # copy, not reference
            'inference_ms': round(inference_ms, 2),
            'frame_index': frame_index,
            'total_frames': self.total_frames,
        }

    def get_model_info(self):
        """Architecture info for the frontend diagram."""
        return self.model.get_architecture_info()

    def cleanup(self):
        """Remove hooks when shutting down."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def benchmark():
    """
    Run a quick benchmark to see if we can hit 30fps.
    No UI, no network — just raw inference speed.
    """
    engine = InferenceEngine()

    print(f"\nbenchmarking 200 frames...")
    times = []

    # warm up — first few passes are always slow
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
