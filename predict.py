"""Run single-image retinal OCT classification with a trained OCTNet checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from models import OCTNet
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.data import build_transforms


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(path_value: str) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line options for single-image inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to an OCT image.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "octnet.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of likely classes to return."
    )
    return parser.parse_args()


def main() -> None:
    """Load one image, print class probabilities, and exit."""
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be at least one")

    config = load_config(args.config)
    image_path = _project_path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Checkpoint does not contain a valid class_names list")

    model = OCTNet(num_classes=len(class_names)).to(device)
    load_checkpoint(checkpoint_path, model, device)
    transform = build_transforms(config["data"])["test"]
    with Image.open(image_path) as image:
        inputs = transform(image.convert("RGB")).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    top_k = min(args.top_k, len(class_names))
    scores, indices = probabilities.topk(top_k)
    predictions = [
        {"class": class_names[index], "probability": round(float(score), 6)}
        for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
    ]
    print(
        json.dumps(
            {"image": str(image_path), "device": str(device), "predictions": predictions},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
