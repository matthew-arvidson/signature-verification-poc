"""
Training script for the signature Siamese network.

This is intentionally lightweight for the POC:
- reads images from `data/samples/<class_name>/`
- builds all positive and negative pairs
- preprocesses each image to 224x224
- trains with contrastive loss
- saves a checkpoint with model config and class metadata
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from data_preprocessing import PreprocessConfig, preprocess_signature_pipeline, to_model_input_chw
from model import PretrainedMode, SignatureModelConfig, build_signature_model

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class PairExample:
    """A pair of image paths and whether they come from different classes."""

    left_path: Path
    right_path: Path
    label: float  # 0.0 = same writer/class, 1.0 = different writer/class


class SignaturePairDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Preprocesses signature images and returns deterministic training pairs."""

    def __init__(
        self,
        sample_root: Path,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        self.sample_root = sample_root
        self.preprocess_config = preprocess_config or PreprocessConfig(signature_search_top=0.0)
        self.class_to_paths = self._discover_class_images(sample_root)
        self.pairs = self._build_pairs(self.class_to_paths)
        self._tensor_cache: Dict[Path, Tensor] = {}

    @staticmethod
    def _discover_class_images(sample_root: Path) -> dict[str, list[Path]]:
        classes: dict[str, list[Path]] = {}
        for class_dir in sorted(path for path in sample_root.iterdir() if path.is_dir()):
            image_paths = sorted(
                path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if image_paths:
                classes[class_dir.name] = image_paths

        if len(classes) < 2:
            raise ValueError(
                f"Expected at least two class folders under {sample_root}, found {len(classes)}"
            )

        return classes

    @staticmethod
    def _build_pairs(class_to_paths: dict[str, list[Path]]) -> list[PairExample]:
        pairs: list[PairExample] = []
        class_names = sorted(class_to_paths)

        for class_name in class_names:
            for left_path, right_path in combinations(class_to_paths[class_name], 2):
                pairs.append(PairExample(left_path=left_path, right_path=right_path, label=0.0))

        for left_class, right_class in combinations(class_names, 2):
            for left_path, right_path in product(
                class_to_paths[left_class], class_to_paths[right_class]
            ):
                pairs.append(PairExample(left_path=left_path, right_path=right_path, label=1.0))

        if not pairs:
            raise ValueError("No training pairs could be created from the sample folders")

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_tensor(self, image_path: Path) -> Tensor:
        cached = self._tensor_cache.get(image_path)
        if cached is not None:
            return cached

        gray_224 = preprocess_signature_pipeline(
            image_path,
            config=self.preprocess_config,
            return_binary=False,
        )
        chw = to_model_input_chw(gray_224, num_channels=3, normalize=True)
        tensor = torch.from_numpy(chw).to(dtype=torch.float32)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        self._tensor_cache[image_path] = tensor
        return tensor

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        pair = self.pairs[index]
        left_tensor = self._load_tensor(pair.left_path)
        right_tensor = self._load_tensor(pair.right_path)
        label_tensor = torch.tensor(pair.label, dtype=torch.float32)
        return left_tensor, right_tensor, label_tensor


class ContrastiveLoss(nn.Module):
    """Classic contrastive loss for same/different pair training."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, left_embedding: Tensor, right_embedding: Tensor, label: Tensor) -> Tensor:
        distances = torch.nn.functional.pairwise_distance(left_embedding, right_embedding)
        positive_term = (1.0 - label) * distances.pow(2)
        negative_term = label * torch.clamp(self.margin - distances, min=0.0).pow(2)
        return torch.mean(positive_term + negative_term)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the signature Siamese model")
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path("data/samples"),
        help="Root directory containing class folders such as genuine/ and other/",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoint.pt"),
        help="Path to save the trained checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Contrastive loss margin for different-class pairs",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension produced by the model",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout applied before the projection head",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ResNet convolutional layers and train only the projection head",
    )
    parser.add_argument(
        "--pretrained",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Whether to initialize ResNet-18 with ImageNet weights",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    return parser


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for repeatable runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    criterion: ContrastiveLoss,
    optimizer: Adam,
    device: torch.device,
) -> float:
    """Run one pass over the pair dataset and return average loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for left_batch, right_batch, labels in dataloader:
        left_batch = left_batch.to(device)
        right_batch = right_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        left_embedding, right_embedding = model(left_batch, right_batch)
        loss = criterion(left_embedding, right_embedding, labels)
        loss.backward()
        optimizer.step()

        batch_size = left_batch.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Adam,
    model_config: SignatureModelConfig,
    dataset: SignaturePairDataset,
    args: argparse.Namespace,
    train_losses: list[float],
) -> None:
    """Persist model weights plus enough metadata for inference."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "class_names": sorted(dataset.class_to_paths.keys()),
        "num_pairs": len(dataset),
        "train_losses": train_losses,
        "args": {
            "sample_root": str(args.sample_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "margin": args.margin,
            "seed": args.seed,
        },
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    sample_root = args.sample_root.resolve()
    checkpoint_path = args.checkpoint_path.resolve()

    dataset = SignaturePairDataset(sample_root=sample_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model_config = SignatureModelConfig(
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        pretrained=args.pretrained,  # type: ignore[arg-type]
    )
    model = build_signature_model(
        embedding_dim=model_config.embedding_dim,
        dropout=model_config.dropout,
        freeze_backbone=model_config.freeze_backbone,
        pretrained=model_config.pretrained,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
    )

    print(f"Training on device: {device}")
    print(f"Sample root: {sample_root}")
    print(f"Classes: {', '.join(sorted(dataset.class_to_paths.keys()))}")
    print(f"Training pairs: {len(dataset)}")

    train_losses: list[float] = []
    for epoch_idx in range(args.epochs):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch_idx + 1:02d}/{args.epochs}: loss={avg_loss:.4f}")

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        dataset=dataset,
        args=args,
        train_losses=train_losses,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
