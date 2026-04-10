"""
Model definitions for offline signature verification.

The network consumes 224x224 signature crops and projects them into an
L2-normalized embedding space so same-writer signatures cluster together
while different-writer signatures remain farther apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


PretrainedMode = Literal["imagenet", "none"]


@dataclass(frozen=True)
class SignatureModelConfig:
    """Config for the signature embedding network."""

    embedding_dim: int = 128
    dropout: float = 0.10
    freeze_backbone: bool = False
    pretrained: PretrainedMode = "imagenet"


def _resolve_resnet18_weights(pretrained: PretrainedMode) -> ResNet18_Weights | None:
    """Resolve torchvision weights enum from a small string API."""
    if pretrained == "imagenet":
        return ResNet18_Weights.DEFAULT
    if pretrained == "none":
        return None
    raise ValueError(f"Unsupported pretrained mode: {pretrained}")


class SignatureEncoder(nn.Module):
    """
    ResNet-18 based encoder that outputs normalized signature embeddings.

    Inputs are expected to be BCHW tensors with shape [batch, 3, 224, 224].
    """

    def __init__(self, config: SignatureModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or SignatureModelConfig()

        weights = _resolve_resnet18_weights(self.config.pretrained)
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features

        # Keep the convolutional trunk and replace the classification head.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(feature_dim, self.config.embedding_dim),
        )

        if self.config.freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the ResNet trunk."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode a batch of signature images into L2-normalized embeddings.

        Args:
            x: Float tensor of shape [batch, 3, 224, 224].

        Returns:
            Tensor of shape [batch, embedding_dim].
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)


class SignatureSiameseNet(nn.Module):
    """Thin Siamese wrapper around a shared signature encoder."""

    def __init__(self, config: SignatureModelConfig | None = None) -> None:
        super().__init__()
        self.encoder = SignatureEncoder(config=config)

    def forward_once(self, x: Tensor) -> Tensor:
        """Encode a single branch."""
        return self.encoder(x)

    def forward(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        """Encode two batches with shared weights."""
        return self.forward_once(left), self.forward_once(right)

    @torch.inference_mode()
    def pairwise_distance(self, left: Tensor, right: Tensor) -> Tensor:
        """Compute Euclidean distance between paired embeddings."""
        left_emb, right_emb = self.forward(left, right)
        return F.pairwise_distance(left_emb, right_emb)


def build_signature_model(
    embedding_dim: int = 128,
    dropout: float = 0.10,
    freeze_backbone: bool = False,
    pretrained: PretrainedMode = "imagenet",
) -> SignatureSiameseNet:
    """Convenience factory for training and inference scripts."""
    config = SignatureModelConfig(
        embedding_dim=embedding_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
    )
    return SignatureSiameseNet(config=config)
