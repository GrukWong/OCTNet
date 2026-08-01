# Examples

Assuming the dataset is in `data/RetinalOCT_Dataset/`:

```bash
python train.py --output-dir outputs/experiment_01
python evaluate.py --checkpoint outputs/experiment_01/best.pt
python predict.py --checkpoint outputs/experiment_01/best.pt --image example.png
```

For a dataset outside the repository, add `--data-root /path/to/RetinalOCT_Dataset` to `train.py` and `evaluate.py`.
