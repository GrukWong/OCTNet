# Reproducibility

## What Is Controlled

`train.py` stores a random seed, effective YAML configuration, training history, log, plot, and best checkpoint for every run. The default configuration uses seed 42 and exposes all original hyperparameters in `configs/octnet.yaml`.

Use deterministic execution when a stricter comparison is needed:

```yaml
training:
  seed: 42
  deterministic: true
```

Deterministic CUDA kernels can reduce throughput and may still differ across PyTorch, CUDA, driver, or hardware versions.

## Original Result Caveat

The archived notebook did not set a seed, record package versions, or save its checkpoint. Therefore, the 97.04% test result is documented as the original reported result, not as a bitwise-reproducible release baseline. A future release should publish the corresponding `best.pt` plus environment metadata.

## Platform Notes

- CUDA uses AMP and `pin_memory`; CPU runs disable AMP safely.
- On Windows, reduce `loader.num_workers` to `0` if multiprocessing causes startup issues. The loader automatically disables worker-only options then.
- Keep class directory names identical across `train`, `val`, and `test`; the pipeline rejects inconsistent ImageFolder mappings.
