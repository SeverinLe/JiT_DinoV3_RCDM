"""
rcdm/encoders/resnet50.py

Frozen ImageNet-supervised ResNet-50 — the control condition.

This is RCDM's original encoder (Bordes et al. 2022), kept so the study has a
*supervised, out-of-domain* baseline against the two self-supervised retinal
encoders.  Bordes et al. found supervised representations invert far less
faithfully than SSL ones (MRR 0.69 vs 0.97-0.99 on ImageNet); reproducing that
contrast on retinal data is what makes the SSL-vs-supervised claim testable
here rather than assumed.

The representation is the 2048-dim global-average-pooled trunk output, obtained
by replacing the classification head with Identity.
"""

import torch
import torch.nn as nn

from .transforms import build_transform  # noqa: F401  (re-exported for callers)

# Width of the ResNet-50 avgpool output.
H_DIM = 2048


class ResNet50Encoder(nn.Module):
    """Frozen ResNet-50 trunk exposing a uniform ``forward(x) -> (B, 2048)``."""

    h_dim = H_DIM
    name = "resnet50"

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) ImageNet-normalised tensor.
        Returns:
            h: (B, 2048) avgpool trunk representation.
        """
        return self.backbone(x)


def load(device: str = "cpu", **_) -> ResNet50Encoder:
    """
    Load the ImageNet-supervised ResNet-50 trunk from torchvision.

    Unlike the other two encoders this downloads its weights from torchvision's
    model zoo on first use, then caches them in ~/.cache/torch.

    Returns:
        ResNet50Encoder in eval mode with all parameters frozen.
    """
    from torchvision import models

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()  # drop the 1000-way head -> (B, 2048)

    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    encoder = ResNet50Encoder(backbone)
    return encoder.eval().to(device)
