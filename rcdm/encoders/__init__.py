"""
rcdm.encoders — the frozen encoders under study.

Every experiment in this project treats the encoder as the independent variable,
so all three are loaded through one registry with one interface:

    from rcdm.encoders import get_encoder, build_transform

    encoder = get_encoder("dinov3", device="cuda")   # frozen, eval mode
    h = encoder(transform(img).unsqueeze(0).to("cuda"))   # (1, encoder.h_dim)

Contract every encoder satisfies:

    encoder.name    str    registry key, written into result manifests
    encoder.h_dim   int    width of h (384 / 1024 / 2048)
    encoder(x)      (B, 3, 224, 224) -> (B, h_dim), no grad, eval mode

Adding an encoder means adding a module with ``H_DIM`` and ``load(device, ...)``
and one line in ENCODERS below — no changes anywhere else in the pipeline, which
is parametrised by ``h_dim`` throughout.
"""

from typing import Callable, Dict

import torch
import torch.nn as nn
from PIL import Image

from . import dinov3, resnet50, retfound
from .transforms import (  # noqa: F401  (public re-exports)
    ENCODER_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transform,
)

# name -> (loader, h_dim, one-line description used in --help and manifests)
ENCODERS: Dict[str, tuple] = {
    "dinov3": (
        dinov3.load,
        dinov3.H_DIM,
        "DinoV3 ViT-S/16, joint-embedding SSL, 384-dim CLS",
    ),
    "retfound_cfp": (
        retfound.load,
        retfound.H_DIM,
        "RETFound MAE ViT-L/16 (colour fundus weights), reconstructive SSL, 1024-dim CLS",
    ),
    "resnet50": (
        resnet50.load,
        resnet50.H_DIM,
        "ImageNet-supervised ResNet-50, 2048-dim avgpool (control)",
    ),
}

ENCODER_NAMES = tuple(ENCODERS)


def get_encoder(name: str, device: str = "cpu", **kwargs) -> nn.Module:
    """
    Load a frozen encoder by registry name.

    Args:
        name: one of ENCODER_NAMES.
        device: "cpu", "cuda" or "mps".
        **kwargs: forwarded to the encoder's loader, e.g. checkpoint_path.

    Returns:
        A frozen nn.Module in eval mode with ``.name`` and ``.h_dim`` set.
    """
    if name not in ENCODERS:
        raise ValueError(
            f"Unknown encoder {name!r}. Available: {', '.join(ENCODER_NAMES)}"
        )
    loader: Callable = ENCODERS[name][0]
    return loader(device=device, **kwargs)


def get_h_dim(name: str) -> int:
    """Representation width for an encoder, without loading its weights."""
    if name not in ENCODERS:
        raise ValueError(
            f"Unknown encoder {name!r}. Available: {', '.join(ENCODER_NAMES)}"
        )
    return ENCODERS[name][1]


def describe(name: str) -> str:
    """One-line human-readable description, for --help text and manifests."""
    return ENCODERS[name][2]


@torch.no_grad()
def encode_image(image_path, encoder: nn.Module, transform, device: str = "cpu") -> torch.Tensor:
    """
    Encode a single image file.

    Returns:
        h: (1, encoder.h_dim)
    """
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    return encoder(x)


@torch.no_grad()
def encode_batch(image_paths, encoder: nn.Module, transform,
                 device: str = "cpu", batch_size: int = 64,
                 verbose: bool = True) -> torch.Tensor:
    """
    Encode a list of image paths in batches.

    Used by data/scripts/precompute_reps.py to cache every representation once,
    so the encoder never runs inside the training loop.

    Returns:
        reps: (N, encoder.h_dim) float32 on CPU, with reps[i] belonging to
        image_paths[i] — index alignment is what the whole pipeline relies on.
    """
    all_reps = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        x = torch.stack(imgs).to(device)
        all_reps.append(encoder(x).float().cpu())
        if verbose and (i // batch_size) % 10 == 0:
            print(f"  encoded {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
    return torch.cat(all_reps, dim=0)
