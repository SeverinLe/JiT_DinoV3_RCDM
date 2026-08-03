"""
precompute_reps.py

Run a frozen encoder over every training image and save representations to disk.

Supports two encoders:
  resnet50  — frozen ResNet-50 backbone → 2048-dim (Tiny ImageNet default)
  retfound  — frozen RETFound ViT-Large → 1024-dim (OCT retinal images)

Usage — Tiny ImageNet (ResNet-50):
    python scripts/precompute_reps.py \
        --data_dir   data/tiny-imagenet-200/train \
        --out_file   data/tiny-imagenet-200/train_reps.pt \
        --image_size 64 \
        --batch_size 128 \
        --device     cpu

Usage — OCT2017 (RETFound ViT-Large):
    python scripts/precompute_reps.py \
        --encoder    retfound \
        --weights    data/RETFound/RETFound_mae_natureOCT.pth \
        --data_dir   data/OCT2017/train \
        --out_file   data/OCT2017/train_reps.pt \
        --image_size 224 \
        --batch_size 32 \
        --device     cuda

Output:
    A .pt file containing a dict:
    {
        "paths"   : list[str]      — image file paths, length N
        "reps"    : Tensor(N, D)   — float32 representations
        "encoder" : str            — "resnet50" or "retfound"
        "h_dim"   : int            — D (2048 or 1024)
    }

    Index alignment is guaranteed: reps[i] belongs to paths[i].
"""

import argparse
import sys
import os
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rcdm.encoder import load_encoder, load_retfound_encoder, build_transform, encode_batch

VALID_EXTENSIONS = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}


def collect_image_paths(data_dir: str) -> list:
    """
    Recursively walk data_dir and return a sorted list of image paths.

    Handles both Tiny ImageNet layout (train/class/images/*.JPEG) and
    standard ImageFolder layout (train/class/*.jpeg) used by OCT2017.
    """
    data_dir = Path(data_dir)
    paths = sorted(
        str(p) for p in data_dir.rglob("*")
        if p.suffix in VALID_EXTENSIONS
    )
    print(f"Found {len(paths)} images in {data_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Precompute encoder representations for RCDM training"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet50",
        choices=["resnet50", "retfound"],
        help="Which encoder to use: resnet50 (2048-dim) or retfound (1024-dim)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to encoder weights file (required for --encoder retfound, "
             "e.g. data/RETFound/RETFound_mae_natureOCT.pth)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/tiny-imagenet-200/train",
        help="Root directory of the training split (ImageFolder layout)",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/tiny-imagenet-200/train_reps.pt",
        help="Path to save the output .pt file",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Resize/crop images to this size before encoding. "
             "Use 64 for Tiny ImageNet, 224 for RETFound.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Images per encoder forward pass. Reduce if OOM. "
             "Recommended: 128 for ResNet-50, 32 for RETFound on GPU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )
    args = parser.parse_args()

    if args.encoder == "retfound" and args.weights is None:
        parser.error("--weights is required when using --encoder retfound")

    # ------------------------------------------------------------------ #
    # Step 1 — collect all image paths
    # ------------------------------------------------------------------ #
    print("\n[1/3] Collecting image paths...")
    paths = collect_image_paths(args.data_dir)

    if len(paths) == 0:
        raise RuntimeError(
            f"No images found in {args.data_dir}. "
            "Check that the path is correct and images have a recognised extension."
        )

    # ------------------------------------------------------------------ #
    # Step 2 — load encoder and run over all images
    # ------------------------------------------------------------------ #
    print(f"\n[2/3] Loading {args.encoder} encoder on {args.device}...")

    if args.encoder == "resnet50":
        encoder = load_encoder(device=args.device)
        h_dim = 2048
    else:
        encoder = load_retfound_encoder(
            weights_path=args.weights,
            device=args.device,
        )
        h_dim = 1024

    transform = build_transform(image_size=args.image_size)

    print(f"Running encoder over {len(paths)} images "
          f"(batch_size={args.batch_size}, h_dim={h_dim})...")

    reps = encode_batch(
        image_paths=paths,
        encoder=encoder,
        transform=transform,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"\nRepresentations shape : {reps.shape}")
    print(f"Representations dtype : {reps.dtype}")
    print(f"Sample norm (first 5) : {reps[:5].norm(dim=1).tolist()}")

    # ------------------------------------------------------------------ #
    # Step 3 — save to disk
    # ------------------------------------------------------------------ #
    print(f"\n[3/3] Saving to {args.out_file}...")
    os.makedirs(Path(args.out_file).parent, exist_ok=True)

    torch.save(
        {
            "paths":   paths,
            "reps":    reps,
            "encoder": args.encoder,
            "h_dim":   h_dim,
        },
        args.out_file,
    )

    # Sanity check
    loaded = torch.load(args.out_file, weights_only=False)
    assert len(loaded["paths"]) == loaded["reps"].shape[0], \
        "Path count and rep count don't match"
    assert loaded["reps"].shape[1] == h_dim, \
        f"Expected {h_dim}-dim reps, got {loaded['reps'].shape[1]}"

    print(f"\nDone. Saved {len(paths)} representations to {args.out_file}")
    print(f"File size : {Path(args.out_file).stat().st_size / 1e6:.1f} MB")
    print("Verification passed — paths and reps are aligned.")


if __name__ == "__main__":
    main()