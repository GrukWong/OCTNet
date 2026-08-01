# Checkpoints

Model weights are not stored in Git. Run `python train.py` to create
`outputs/train/best.pt`, then supply that path to `evaluate.py` or `predict.py`.

If release weights are published later, place them in this directory or attach
them to a GitHub Release rather than committing them to the repository.
