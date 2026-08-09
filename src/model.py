"""
Sinusoidal MLP for encoding video as a neural function.

The idea: instead of storing video as pixels, we teach a small neural net
the function f(t, x, y) -> brightness. Feed it any coordinate, it gives
you the pixel value. The entire video lives in the weights.

Architecture based on Sitzmann et al. 2020 "Implicit Neural Representations
with Periodic Activation Functions" — but we don't need to be academic about it.
It's just sin() activations with careful weight initialization.
"""

import torch
import torch.nn as nn
import numpy as np
import math


class SineLayer(nn.Module):
    """
    A single linear layer followed by sin() activation.

    The magic is in the initialization — this won't converge properly
    without it. Trust the math here, I've tried random init and it
    produces garbage blurry frames.

    omega_0 controls the "frequency" of the sine. Higher = more detail
    but harder to train. 30 works well for video.
    """

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        # first layer and hidden layers need different init ranges.
        # this comes from the sinusoidal representation paper — eq. 9 and 10.
        # messing with this will break training, ask me how i know.
        with torch.no_grad():
            n = self.linear.in_features
            if self.is_first:
                bound = 1.0 / n
            else:
                bound = math.sqrt(6.0 / n) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        # that's it. that's the whole layer. linear -> scale -> sin.
        return torch.sin(self.omega_0 * self.linear(x))


class SineMLP(nn.Module):
    """
    Full sinusoidal MLP: stack of SineLayers + linear output.

    Input:  (t, x, y) — 3 values, all in [-1, 1]
    Output: pixel brightness — 1 value in [0, 1]

    Default config (5 layers, 256 wide) gives ~264K params.
    That's small enough to run inference at 30fps on an RTX 2050
    while still capturing enough detail for a watchable video.
    """

    def __init__(self, in_features=3, out_features=1,
                 hidden_features=256, hidden_layers=5, omega_0=30.0):
        super().__init__()

        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers

        layers = []

        # first layer — takes the raw (t, x, y) coordinates
        layers.append(SineLayer(in_features, hidden_features,
                                omega_0=omega_0, is_first=True))

        # hidden layers — this is where the detail gets learned
        for i in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features,
                                    omega_0=omega_0))

        self.net = nn.Sequential(*layers)

        # output layer — plain linear, no sine.
        # sigmoid squashes to [0, 1] which is what we need for pixel values.
        self.output_layer = nn.Linear(hidden_features, out_features)

        # init output layer the same way as hidden layers
        with torch.no_grad():
            n = hidden_features
            bound = math.sqrt(6.0 / n) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)

    def forward(self, x):
        h = self.net(x)
        # Return raw logits for BCEWithLogitsLoss.
        # The inference engine will apply sigmoid.
        out = self.output_layer(h)
        return out

    def get_architecture_info(self):
        """
        Returns a dict describing the network shape — used by the frontend
        to draw the node-link diagram. Called once when the page loads.
        """
        info = {
            'input_dim': 3,
            'input_labels': ['t', 'x', 'y'],
            'output_dim': 1,
            'output_labels': ['pixel'],
            'hidden_features': self.hidden_features,
            'hidden_layers': self.hidden_layers,
            'layer_names': [],
            'total_params': sum(p.numel() for p in self.parameters()),
        }

        # grab the name of each sine layer — these are the ones we hook
        for name, module in self.net.named_modules():
            if isinstance(module, SineLayer):
                info['layer_names'].append(f'net.{name}')

        return info


def build_model(device='cpu'):
    """Quick helper to build and move to device."""
    
    # Oops. The README used to brag about 1.8 million parameters, but 
    # someone on GitHub correctly pointed out I hardcoded this to 256/5 here.
    # It turns out the model only has 264K parameters! 
    # The fact that it still memorized the video this well with 7x fewer 
    # params is a total flex, so we're keeping it this way.
    model = SineMLP(
        in_features=3,
        out_features=1,
        hidden_features=256,
        hidden_layers=5,
        omega_0=30.0
    )
    return model.to(device)


if __name__ == '__main__':
    # sanity check — make sure shapes work and print param count
    model = build_model()
    info = model.get_architecture_info()

    print(f"model built")
    print(f"  layers: {info['hidden_layers']} hidden, {info['hidden_features']} wide")
    print(f"  params: {info['total_params']:,}")
    print(f"  layer names: {info['layer_names']}")

    # test forward pass with dummy data
    dummy = torch.randn(100, 3)  # 100 random (t,x,y) coords
    out = model(dummy)
    print(f"  test forward: input {dummy.shape} -> output {out.shape}")
    print(f"  output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
