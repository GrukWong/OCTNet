# Method

## Architecture

`OCTNet` is a custom five-stage CNN that processes RGB images resized to 224×224. 

| Stage | Input → output channels | SE attention | Downsampling |
| --- | --- | --- | --- |
| 1 | 3 → 32 | No | 2×2 MaxPool |
| 2 | 32 → 64 | No | 2×2 MaxPool |
| 3 | 64 → 128 | Yes | 2×2 MaxPool |
| 4 | 128 → 256 | Yes | 2×2 MaxPool |
| 5 | 256 → 512 | Yes | 2×2 MaxPool |

When a stage changes the channel count, a 1×1 convolution projects the input. The resulting tensor passes through a residual block:

```text
x → 7×7 depthwise Conv → GroupNorm(1 group) → 1×1 Conv (4× expansion)
  → GELU → 1×1 Conv → +x
```

Stages 3–5 then apply squeeze-and-excitation (SE) channel reweighting. Global average pooling feeds a `512 → 1024 → 8` classifier with GELU and dropout (`p=0.5`). The original measurement reports 3.60M parameters and 2.32G FLOPs.

## Preprocessing and Augmentation

All images are resized with bicubic interpolation and normalized with mean `[0.210, 0.210, 0.210]` and standard deviation `[0.182, 0.182, 0.182]`.

Training additionally uses:

1. random horizontal flip;
2. random rotation in ±15 degrees;
3. brightness and contrast jitter (0.2 each);
4. Random Erasing with probability 0.4.

Validation, test, and single-image inference use only resize and normalization.

## Optimization

The training objective is cross entropy with label smoothing 0.1. Optimization uses AdamW with learning rate `5e-4` and weight decay `3e-5`. A cosine annealing schedule runs for 20 epochs (`eta_min=1e-6`). CUDA training uses automatic mixed precision; CPU execution falls back to full precision.

The best checkpoint is selected by validation Accuracy, matching the original experiment's stated model-selection criterion.
