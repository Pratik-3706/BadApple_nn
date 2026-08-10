# BadApple × NN
A neural network that memorizes the entire Bad Apple music video and regenerates it live, frame by frame, from pure math. No video file, no frames on disk just ~1,050,000 learned numbers (weights) and a function that takes in `(time, x, y)` and spits out a pixel brightness.

You give it a coordinate. It gives you a pixel. Do that 172,800 times and you get a frame. Do *that* 6,505 times and you get the whole video. The entire 3:39 music video lives inside a ~4.02MB file.
---

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue?style=for-the-badge)](https://huggingface.co/Pratik-3706/bad_epoch)

Download the pre-trained neural network weights instantly from Hugging Face and run the web player without training it yourself!

## License & Copyright
- **The Code**: All original code in this repository is released under the **MIT License**.
- **The Model & Assets**: The "Bad Apple!!" video, audio, and the neural network weights (which are a derivative mathematical representation of the video) do **not** belong to me. They are the intellectual property of ZUN / Team Shanghai Alice and Alstroemeria Records. The weights are provided strictly for educational and transformative AI research purposes under Fair Use. You cannot use the model for commercial purposes.

## wait, what?

Yeah so the idea is actually kind of stupid-simple:

Instead of storing the video as a giant pile of pixel data, we teach a tiny neural network the *function* behind the video. Mathematically, Bad Apple is just a function:

```
f(time, x, y) → brightness
```

A black pixel at the top-left corner of frame 1000? That's just `f(0.31, -1.0, -1.0) → 0.0`. A white pixel in the center of frame 3000? That's `f(0.92, 0.0, 0.0) → 1.0`. Every single pixel is just a point on this function.

So we train a neural network to *become* that function. And then we throw away the video entirely.

## how it actually works

The network uses **sinusoidal activations** (SIREN architecture). Instead of the usual ReLU activations that most neural nets use, every layer passes its output through `sin()`. This lets the network learn extremely sharp, high-frequency patterns which is exactly what you need to reconstruct crispy black-and-white edges.

The architecture:
- **Input**: 3 values - `(t, x, y)`, all normalized to `[-1, 1]`
- **Hidden layers**: 5 layers, 512 neurons each, all with `sin()` activations
- **Output**: 1 value - pixel brightness (0 = black, 1 = white)
- **omega_0**: 30.0 (controls how "detailed" the sine waves can get)
- **Total params**: ~1.05M

Training uses `BCEWithLogitsLoss` instead of the more common `MSELoss` because Bad Apple is essentially a binary video pixels are either black or white. BCE brutally punishes the network for predicting wishy-washy gray values, which forces it to commit to razor-sharp edges.

## the web player

Once trained, a FastAPI server runs the neural network live and streams the generated frames to a web browser over WebSocket. The main panel shows the video being generated in real-time. The bottom panel shows a live heatmap of every single neuron in every layer. To keep the visualization dynamic and fast, this heatmap specifically tracks the exact mathematical activations for the **dead-center pixel** of the video. You are watching 2,560 sine waves oscillate in real-time as the network calculates the center of the screen!

The audio plays synced in the browser while the neural network generates the visuals at 29.9 fps. The inference engine natively utilizes FP16 precision on GPUs to trigger Tensor Cores, enabling perfectly smooth real-time generation even on massive frame grids. If the GPU still can't keep up, it'll drop frames to stay in sync rather than going slow-mo.

## running it yourself

### prerequisites

- Python 3.10+
- An NVIDIA GPU with CUDA (I tortured an RTX 2050 for this. If yours is better, it will simply suffer less)
- The Bad Apple video (put it in `bad_apple_vid/vid.mp4`)

### setup

```bash
# create a venv (optional but recommended)
python -m venv venv
venv\Scripts\activate        # windows
# source venv/bin/activate   # linux/mac

# install dependencies
pip install -r requirements.txt
```

### step 1: extract frames

This rips every frame out of the video, converts them to grayscale, resizes to 480×360, and packs them into a single `.npy` file.

```bash
python src/extract_frames.py
```

### step 2: train

This is the long part. On an RTX 2050 it takes a few hours. The script saves checkpoints automatically whenever the loss improves, so you can kill it and come back later it picks up right where it left off.

```bash
python src/train.py
```

You can peek at `train_outputs/` to see sample frames as training progresses. They start out as blurry ghosts and slowly sharpen into recognizable frames.

### step 3: watch it live

```bash
python src/server.py
```

Then open `http://localhost:8000` in your browser and hit play.

## project structure

```
src/
  model.py           - the SIREN neural network architecture
  dataset.py         - loads frames, builds coordinate grids, serves random batches
  train.py           - training loop with AMP, checkpoint saving/loading
  extract_frames.py  - rips frames from the video into a numpy array
  inference_engine.py- runs the trained model live, captures layer activations
  server.py          - FastAPI + WebSocket server for real-time streaming
  evaluate.py        - SSIM/PSNR quality metrics

static/
  index.html         - the web player UI
  app.js             - frontend logic, network diagram rendering, activation viz
  style.css          - dark theme styling

bad_apple_vid/       - put your vid.mp4 here
checkpoints/         - trained model weights (.pt)
data/                - extracted frames (.npy)
train_outputs/       - sample frames saved during training
```

## things i learned building this

1. **Weight initialization matters more than you think.** Random init with sine activations produces absolute garbage. The SIREN paper's init scheme (equations 9 and 10) is non-negotiable.

2. **MSE loss makes everything foggy.** Switching to `BCEWithLogitsLoss` was the single biggest improvement for getting clean black/white separation.

3. **omega_0 is the sharpness dial.** Too low (10-20) and everything is soft and blurry. Too high (100+) and training becomes unstable. 30 is the sweet spot for this video.

4. **AMP (mixed precision) is free speed.** Literally just wrap your forward pass in `torch.cuda.amp.autocast()` and training goes ~2x faster with zero quality loss.

5. **You don't need to train on every pixel every epoch.** Capping steps per epoch and just randomly sampling pixels works just as well and lets you checkpoint way more often.

## credits

- **Bad Apple!!** - Original song from the Touhou Project by ZUN. Shadow art video by あにら (Anira) on Nico Nico Douga.
- **SIREN** "Implicit Neural Representations with Periodic Activation Functions" by Sitzmann et al. 2020
- Built in the proud tradition of "will it play Bad Apple?" right next to oscilloscopes, graphing calculators, and Minecraft redstone.

## license

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (or treat it as MIT). Feel free to fork, modify, and build your own Bad Apple neural networks!
