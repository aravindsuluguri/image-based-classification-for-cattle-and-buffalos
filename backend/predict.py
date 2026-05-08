"""Inference helper for the cattle breed model.

Usage as a module:
    from predict import predict_breed
    result = predict_breed("/path/to/image.jpg")
    # or
    result = predict_breed("https://example.com/cow.jpg")

Usage from CLI:
    python backend/predict.py --source /path/to/image.jpg
    python backend/predict.py --source https://example.com/cow.jpg --top-k 3
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import timm
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

import gc
import os
import gdown

MODEL_PATH = Path(__file__).resolve().parent / "best_model.pth"

if not MODEL_PATH.exists():
    url = "https://drive.google.com/file/d/1ecnLPKhAqnXGprg3A_smy_Il_GE42_Wr/view?usp=drive_link"
    gdown.download(url, str(MODEL_PATH), quiet=False)

DEFAULT_MODEL_PATH = MODEL_PATH
DEFAULT_METADATA_PATH = Path(__file__).resolve().parent / "metadata.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class InvalidImageSourceError(ValueError):
    """Raised when the source does not provide a decodable image."""


class BreedClassifier(nn.Module):
    """Model architecture used during training."""

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        pretrained: bool = False,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _load_image(source: str | Path) -> Image.Image:
    source_str = str(source)
    if _is_url(source_str):
        try:
            response = requests.get(
                source_str,
                timeout=20,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                },
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise InvalidImageSourceError(f"Failed to fetch image URL: {exc}") from exc

        # Try to open as image regardless of Content-Type header
        try:
            return Image.open(BytesIO(response.content)).convert("RGB")
        except (UnidentifiedImageError, Exception) as exc:
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                raise InvalidImageSourceError(
                    "The URL points to a web page, not a direct image. Try right-clicking the image and copying the image address."
                ) from exc
            raise InvalidImageSourceError(
                "The downloaded content could not be decoded as an image."
            ) from exc

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        return Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise InvalidImageSourceError("The selected file is not a readable image.") from exc


def _load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_preprocess(img_size: int) -> transforms.Compose:
    resize_size = int(round(img_size * 1.1))
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def _load_model(
    model_path: str | Path,
    metadata: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = BreedClassifier(
        num_classes=int(metadata["num_classes"]),
        model_name=str(metadata["model_name"]),
        pretrained=False,
    )
    checkpoint = torch.load(Path(model_path).expanduser().resolve(), map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_breed(
    source: str | Path,
    top_k: int = 5,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    use_cpu: bool = False,
) -> dict[str, Any]:
    """Predict breed from a local image path or URL.

    Args:
        source: Local file path or HTTP/HTTPS image URL.
        top_k: Number of highest-confidence classes to return.
        model_path: Path to .pth checkpoint.
        metadata_path: Path to metadata.json.
        use_cpu: Force CPU inference.

    Returns:
        Prediction dictionary with top result and top-k list.
    """

    metadata = _load_metadata(metadata_path)
    class_names: list[str] = metadata["class_names"]
    breeds_info: dict[str, dict[str, str]] = metadata.get("breeds_info", {})

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    top_k = min(top_k, len(class_names))

    device = torch.device("cpu")
    model = _load_model(model_path=model_path, metadata=metadata, device=device)
    preprocess = _build_preprocess(int(metadata["img_size"]))

    image = _load_image(source)

    # Test-Time Augmentation (TTA) — average predictions from multiple views
    img_size = int(metadata["img_size"])
    resize_size = int(round(img_size * 1.1))

    tta_transforms = [
        # Original center crop
        transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Slight zoom (tighter crop)
        transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(int(img_size * 0.9)),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Slight rotation
        transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(img_size),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]

    with torch.no_grad():
        all_probs = []
        for tta_t in tta_transforms:
            input_tensor = tta_t(image).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            all_probs.append(probs)
        # Average across all TTA views
        avg_probs = torch.stack(all_probs).mean(dim=0)

    top_probs, top_indices = torch.topk(avg_probs, k=top_k)

    predictions: list[dict[str, Any]] = []
    for score, idx in zip(top_probs.tolist(), top_indices.tolist()):
        breed_name = class_names[idx]
        info = breeds_info.get(breed_name, {})
        predictions.append(
            {
                "breed": breed_name,
                "confidence": float(score),
                "type": info.get("type", "unknown"),
                "description": info.get("description", ""),
            }
        )

    return {
        "source": str(source),
        "top_prediction": predictions[0],
        "top_k": predictions,
        "device": str(device),
        "model_name": metadata.get("model_name"),
    }
gc.collect()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict cattle breed from image path or URL")
    parser.add_argument("--source", required=True, help="Image local path or HTTP/HTTPS URL")
    parser.add_argument("--top-k", type=int, default=5, help="How many predictions to return")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to best_model.pth")
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH), help="Path to metadata.json")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = predict_breed(
        source=args.source,
        top_k=args.top_k,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        use_cpu=args.cpu,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()








# how to predict 
# from predict import predict_breed
# result = predict_breed("/absolute/path/to/photo.jpg")
# or
# result = predict_breed("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSyCrIzJTrDfBkZMg2MsaoHXsaMqJp89MdTLX-LuGtwbJdrj_t3P-L38dSQvumxpGKjVJ3BXWNIT-R1g2V5QvKNMk3pacPDlDXxR_Fk15TVfw&s=10")
# print(result["top_prediction"])
# print(result["top_k"])

# or in commandline

# python3 "cattle updated/backend/predict.py" --source "/absolute/path/to/photo.jpg"
# python3 "cattle updated/backend/predict.py" --source "https://www.google.com/imgres?q=Hariana%20indian%20cow&imgurl=https%3A%2F%2Fwww.apnikheti.com%2Fupload%2FliveStock%2F6228idea99hariaana.jpg&imgrefurl=https%3A%2F%2Fwww.apnikheti.com%2Fen%2Fpn%2Flivestock%2Fcow%2Fhariana&docid=MOndugJMIXPPbM&tbnid=BRlcYdraaIvqKM&vet=12ahUKEwibici2sp6TAxWgRWwGHUbLIuQQnPAOegQIGRAB..i&w=311&h=242&hcb=2&ved=2ahUKEwibici2sp6TAxWgRWwGHUbLIuQQnPAOegQIGRAB" --top-k 3


