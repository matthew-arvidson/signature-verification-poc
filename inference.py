"""
Inference utilities for the signature verifier POC.

Given two signature images, this module:
- loads the trained checkpoint
- preprocesses both images exactly like training
- computes embedding distance
- maps that distance to a simple similarity score and verdict
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch import Tensor

from data_preprocessing import PreprocessConfig, preprocess_signature_pipeline, to_model_input_chw
from model import SignatureModelConfig, build_signature_model

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
ImageInput = str | Path | np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    """Human-friendly signature comparison result."""

    distance: float
    similarity_percent: float
    verdict: str


def preprocess_image_to_tensor(
    image_source: ImageInput,
    preprocess_config: PreprocessConfig | None = None,
) -> Tensor:
    """Load one image and convert it to a normalized BCHW tensor."""
    config = preprocess_config or PreprocessConfig(signature_search_top=0.0)
    gray_224 = preprocess_signature_pipeline(
        image_source,
        config=config,
        return_binary=False,
    )
    chw = to_model_input_chw(gray_224, num_channels=3, normalize=True)
    tensor = torch.from_numpy(chw).to(dtype=torch.float32)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0)


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    """Load model weights and metadata from a saved training checkpoint."""
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)

    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint is missing a valid model_config")

    model_config = SignatureModelConfig(**raw_config)
    model = build_signature_model(
        embedding_dim=model_config.embedding_dim,
        dropout=model_config.dropout,
        freeze_backbone=model_config.freeze_backbone,
        pretrained="none",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(resolved_device)
    model.eval()
    return model, checkpoint, resolved_device


def distance_to_similarity_percent(distance: float) -> float:
    """
    Convert Euclidean embedding distance to a 0-100 similarity score.

    Embeddings are L2-normalized, so pairwise distance is typically in [0, 2].
    """
    normalized = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
    return normalized * 100.0


def classify_distance(
    distance: float,
    match_threshold: float,
    review_threshold: float,
) -> str:
    """Translate raw distance into a simple demo verdict."""
    if review_threshold < match_threshold:
        raise ValueError("review_threshold must be greater than or equal to match_threshold")
    if distance <= match_threshold:
        return "match"
    if distance <= review_threshold:
        return "review"
    return "mismatch"


@torch.inference_mode()
def compare_signatures(
    left_image_path: ImageInput,
    right_image_path: ImageInput,
    checkpoint_path: Path = Path("checkpoint.pt"),
    match_threshold: float = 0.45,
    review_threshold: float = 0.75,
    device: torch.device | None = None,
) -> InferenceResult:
    """Compare two signature images using the trained Siamese encoder."""
    model, _checkpoint, resolved_device = load_checkpoint_model(checkpoint_path, device=device)
    left_tensor = preprocess_image_to_tensor(left_image_path).to(resolved_device)
    right_tensor = preprocess_image_to_tensor(right_image_path).to(resolved_device)

    left_embedding, right_embedding = model(left_tensor, right_tensor)
    distance = float(torch.nn.functional.pairwise_distance(left_embedding, right_embedding).item())
    similarity_percent = distance_to_similarity_percent(distance)
    verdict = classify_distance(distance, match_threshold, review_threshold)
    return InferenceResult(
        distance=distance,
        similarity_percent=similarity_percent,
        verdict=verdict,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two signature images")
    parser.add_argument("left_image", type=Path, help="Reference signature image path")
    parser.add_argument("right_image", type=Path, help="Questioned signature image path")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoint.pt"),
        help="Path to a saved training checkpoint",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.45,
        help="Distance at or below this value is treated as a match",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.75,
        help="Distance at or below this value is treated as review; above is mismatch",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    result = compare_signatures(
        left_image_path=args.left_image.resolve(),
        right_image_path=args.right_image.resolve(),
        checkpoint_path=args.checkpoint_path.resolve(),
        match_threshold=args.match_threshold,
        review_threshold=args.review_threshold,
    )

    print(f"Distance: {result.distance:.4f}")
    print(f"Similarity: {result.similarity_percent:.1f}%")
    print(f"Verdict: {result.verdict}")


if __name__ == "__main__":
    main()
