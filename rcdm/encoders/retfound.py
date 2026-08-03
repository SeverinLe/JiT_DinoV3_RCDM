"""
rcdm/encoders/retfound.py

Frozen RETFound MAE ViT-Large/16 encoder — reconstructive (MAE) SSL, pre-trained
on 1.6 M retinal images (Zhou et al., Nature 2023).

Why this encoder is in the study
--------------------------------
RETFound is the domain foundation model: it is what a practitioner would
actually freeze and linear-probe for a retinal grading task.  It represents the
*reconstructive* branch of SSL, against DinoV3's *joint-embedding* branch.

Two weight sets exist and they are not interchangeable:

    RETFound_mae_natureCFP.pth   colour fundus photography  <- used here
    RETFound_mae_natureOCT.pth   optical coherence tomography

Both are ViT-L/16 with a 1024-dim CLS token; only the pre-training corpus
differs.  The report must state which one produced each result.

Why the state dict is filtered
------------------------------
The .pth is a raw MAE *pre-training* checkpoint: it holds the ViT-L encoder
(cls_token, pos_embed, patch_embed, blocks.0-23, norm) *and* the MAE decoder
(every decoder_* key, plus mask_token).  The decoder only reconstructs masked
patches during pre-training and has no counterpart in timm's encoder-only
vit_large_patch16_224.  Dropping those keys before load_state_dict(strict=False)
leaves zero missing/unexpected keys for the encoder itself — which
``load()`` verifies rather than assumes.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .transforms import build_transform  # noqa: F401  (re-exported for callers)

# CLS-token width of ViT-Large/16.
H_DIM = 1024

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = _REPO_ROOT / "encoders" / "retfound_cfp" / "RETFound_mae_natureCFP.pth"

# MAE-decoder-only prefixes absent from the timm encoder.
_MAE_DECODER_PREFIXES = (
    "decoder_embed",
    "decoder_blocks",
    "decoder_norm",
    "decoder_pred",
    "decoder_pos_embed",
    "mask_token",
)


class RETFoundEncoder(nn.Module):
    """Frozen RETFound ViT-L/16 exposing a uniform ``forward(x) -> (B, 1024)``."""

    h_dim = H_DIM
    name = "retfound_cfp"

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) ImageNet-normalised tensor.
        Returns:
            h: (B, 1024) CLS token.
        """
        # timm's forward_features returns the full token sequence (B, 197, 1024);
        # token 0 is the CLS token.
        tokens = self.vit.forward_features(x)
        return tokens[:, 0]


def load(device: str = "cpu", checkpoint_path=DEFAULT_CHECKPOINT,
         strict_keys: bool = True, **_) -> RETFoundEncoder:
    """
    Load the frozen RETFound MAE ViT-L/16 encoder.

    Args:
        device: "cpu", "cuda" or "mps".
        checkpoint_path: path to RETFound_mae_natureCFP.pth (or the OCT weights).
        strict_keys: if True (default), raise when any *encoder* key is missing
            or unexpected after the MAE-decoder keys have been filtered out.
            Silent key mismatches would mean partially random weights.

    Returns:
        RETFoundEncoder in eval mode with all parameters frozen.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for the RETFound encoder: pip install timm") from exc

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found. RETFound weights are access-gated; "
            "see encoders/README.md."
        )

    # num_classes=0 -> no head; global_pool="" -> no pooling, so forward_features
    # returns the raw token sequence and we choose the CLS token explicitly.
    vit = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    encoder_state = {
        k: v for k, v in state_dict.items()
        if not k.startswith(_MAE_DECODER_PREFIXES)
    }

    result = vit.load_state_dict(encoder_state, strict=False)
    if strict_keys and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(
            "RETFound weights did not load cleanly — some encoder parameters "
            f"would be randomly initialised.\n  missing: {result.missing_keys}\n"
            f"  unexpected: {result.unexpected_keys}\n"
            "Pass strict_keys=False only if you understand the consequence."
        )

    for param in vit.parameters():
        param.requires_grad = False
    vit.eval()

    encoder = RETFoundEncoder(vit)
    return encoder.eval().to(device)
