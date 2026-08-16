# training history

The model didn't start at 1.05M parameters. i trained it in stages, fixing bugs and scaling up as i went.

## first run (256-wide, ~264K params)

Started training at 480×360 resolution with a 5-layer, 256-wide SIREN. By epoch 200 the loss bottomed out at **0.0945**. The silhouettes were recognizable but edges were soft and fast-moving scenes were blurry. The network just didn't have enough capacity to memorize 6,505 frames worth of detail at this size.

## widened the network at epoch 242 (256 → 512)

Instead of retraining from scratch, i used a Net2Net-style transformation: padded all the weight matrices from 256 columns to 512 by copying the old weights into the top-left corner and filling the new neurons with tiny random noise to break symmetry. This preserved everything the 256-wide model had already learned while giving it 4x more capacity to learn finer details.

- **Before**: 5 layers × 256 neurons = ~264K params
- **After**: 5 layers × 512 neurons = ~1.05M params

Training resumed from epoch 242 with the same learning rate schedule.

## epoch 380 — best checkpoint

After the widened model settled in, the edges got noticeably sharper, especially on fast-moving silhouettes that the old 256-wide model couldn't handle. The loss is higher than the 256-wide best (0.2434 vs 0.0945) because the model is now computing sigmoid-corrected outputs instead of raw logits, so the numbers aren't directly comparable. Visually, this is the best checkpoint so far.

## code review found two bugs

Found these after epoch 380. Someone pointed out that my motion-aware sampling code was dead — i computed per-frame weights based on how much each frame changed from the previous one, but `get_batch()` never actually used them. It was just doing uniform random sampling the whole time. On top of that, the inference engine was skipping `sigmoid()` entirely and clipping raw logits directly to pixels, which meant mid-gray values were rendering wrong.

i fixed both:
- wired up the frame weights so high-motion frames get sampled more during training
- added `sigmoid(logits * 4.0)` to the inference path (the `* 4.0` keeps the sharp contrast i wanted while fixing the math)

Training will resume with the corrected code.
