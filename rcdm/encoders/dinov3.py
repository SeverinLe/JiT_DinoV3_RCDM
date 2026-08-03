"""
rcdm/encoders/dinov3.py

Frozen DinoV3 ViT-S/16 encoder — joint-embedding (DINO-style) SSL.

Origin: entirely new relative to the upstream repos.  RCDM loaded a torchvision
ResNet-50 inline in train.py and ran it on every training step; JiT had no image
encoder at all (it was class-conditional).

Why this encoder is in the study
--------------------------------
It represents the *joint-embedding* branch of SSL, as opposed to RETFound's
*reconstructive* (MAE) branch.  DINO-style training pulls together augmented
views, so nearby vectors correspond to perceptually similar images — the
property representation-conditioned generation relies on.  At 384 dimensions it
also keeps the conditioning path light.

The use_gated_mlp trap
----------------------
The local checkpoint was trained with a standard 2-projection FFN (up_proj +
down_proj), but the config.json shipped with `use_gated_mlp: true`, which makes
transformers expect a 3-projection gated FFN.  The missing `gate_proj` is then
randomly initialised *on every load*, producing different h vectors run to run.
`assert_config_sane()` below fails loudly rather than letting that corrupt an
experiment silently.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .transforms import build_transform  # noqa: F401  (re-exported for callers)

# CLS-token width of DinoV3 ViT-S/16.  Architecture constant: ViT-S = 384,
# ViT-B = 768, ViT-L = 1024.
H_DIM = 384

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = _REPO_ROOT / "encoders" / "dinov3_vits16"


class DinoV3Encoder(nn.Module):
    """
    Frozen DinoV3 ViT-S/16 exposing a uniform ``forward(x) -> (B, 384)``.

    The wrapper exists so that every encoder in this package has the same
    interface despite different underlying APIs (transformers here, timm for
    RETFound, torchvision for ResNet-50).
    """

    h_dim = H_DIM
    name = "dinov3"

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) ImageNet-normalised tensor.
        Returns:
            h: (B, 384) CLS token from the final layer.
        """
        out = self.backbone(pixel_values=x)
        return out.last_hidden_state[:, 0, :]


def assert_config_sane(checkpoint_path: Path) -> None:
    """Raise if config.json would trigger random gate_proj initialisation."""
    import json

    config_file = Path(checkpoint_path) / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"{config_file} not found. See encoders/README.md for how to obtain "
            "the DinoV3 ViT-S/16 checkpoint."
        )
    config = json.loads(config_file.read_text())
    if config.get("use_gated_mlp", False):
        raise ValueError(
            f"{config_file} has use_gated_mlp: true. The checkpoint has no "
            "gate_proj weights, so transformers would randomly initialise them "
            "on every load and h would differ between runs. Set it to false."
        )


def load(device: str = "cpu", checkpoint_path=DEFAULT_CHECKPOINT, **_) -> DinoV3Encoder:
    """
    Load the frozen DinoV3 ViT-S/16 backbone.

    Args:
        device: "cpu", "cuda" or "mps".
        checkpoint_path: local HuggingFace-format directory containing
            config.json and model.safetensors.  Loaded with
            local_files_only=True so no network request is ever made — the
            weights are pinned for reproducibility.

    Returns:
        DinoV3Encoder in eval mode with all parameters frozen.
    """
    from transformers import AutoModel

    checkpoint_path = Path(checkpoint_path)
    assert_config_sane(checkpoint_path)

    backbone = AutoModel.from_pretrained(str(checkpoint_path), local_files_only=True)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    encoder = DinoV3Encoder(backbone)
    return encoder.eval().to(device)
