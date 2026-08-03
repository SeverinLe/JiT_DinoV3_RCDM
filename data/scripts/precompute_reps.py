"""
data/scripts/precompute_reps.py

Run a frozen encoder over an image split once and cache the representations.

The encoder is a fixed feature extractor in this project, so running it inside
the training loop (as upstream RCDM did) recomputes identical vectors every
epoch.  Caching once also guarantees that training, sampling and every probe in
experiments/ see byte-identical h for the same image.

Output (.pt):
    {
        "paths"   : list[str]    N image paths, sorted
        "labels"  : list[str]    parent directory name of each path (the class)
        "reps"    : Tensor(N, D) float32
        "encoder" : str          registry name, e.g. "dinov3"
        "h_dim"   : int          D
    }

reps[i] belongs to paths[i] — the whole pipeline depends on that alignment.

Usage
-----
    # DinoV3 (384-dim) over the training split
    python data/scripts/precompute_reps.py \
        --encoder  dinov3 \
        --data_dir data/raw/messidor2/train \
        --out_file data/processed/messidor2/dinov3/train_reps.pt

    # RETFound (1024-dim), all three splits
    for split in train val test; do
        python data/scripts/precompute_reps.py \
            --encoder  retfound_cfp \
            --data_dir data/raw/messidor2/$split \
            --out_file data/processed/messidor2/retfound_cfp/${split}_reps.pt
    done

E5 (the downstream linear probe) needs val and test representations as well as
train, so run all three splits for any encoder you intend to probe.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rcdm.encoders import (  # noqa: E402
    ENCODER_NAMES,
    build_transform,
    describe,
    encode_batch,
    get_encoder,
)

VALID_EXTENSIONS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def collect_image_paths(data_dir: Path) -> list:
    """
    Recursively collect image paths under data_dir, sorted for determinism.

    Handles the ImageFolder layout used by both datasets:
        <split>/<class>/<image>.png
    """
    paths = sorted(
        str(p) for p in Path(data_dir).rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found under {data_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache frozen-encoder representations for a dataset split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Encoders:\n  " + "\n  ".join(f"{n:14s} {describe(n)}" for n in ENCODER_NAMES),
    )
    parser.add_argument("--encoder", required=True, choices=ENCODER_NAMES)
    parser.add_argument("--data_dir", required=True,
                        help="Split root, e.g. data/raw/messidor2/train")
    parser.add_argument("--out_file", required=True,
                        help="Destination .pt, e.g. data/processed/messidor2/dinov3/train_reps.pt")
    parser.add_argument("--encoder_ckpt", default=None,
                        help="Override the encoder weights path (defaults per encoder)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    paths = collect_image_paths(Path(args.data_dir))
    labels = [Path(p).parent.name for p in paths]
    print(f"Found {len(paths)} images in {args.data_dir}")
    print(f"  classes: {sorted(set(labels))}")

    print(f"Loading encoder '{args.encoder}' — {describe(args.encoder)}")
    kwargs = {"checkpoint_path": args.encoder_ckpt} if args.encoder_ckpt else {}
    encoder = get_encoder(args.encoder, device=args.device, **kwargs)

    transform = build_transform()  # always 224 px, ImageNet normalisation
    reps = encode_batch(paths, encoder, transform,
                        device=args.device, batch_size=args.batch_size)

    if reps.shape[0] != len(paths):
        raise RuntimeError(f"Alignment broken: {reps.shape[0]} reps for {len(paths)} paths")
    if reps.shape[1] != encoder.h_dim:
        raise RuntimeError(f"Expected h_dim {encoder.h_dim}, got {reps.shape[1]}")

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "paths": paths,
            "labels": labels,
            "reps": reps,
            "encoder": args.encoder,
            "h_dim": encoder.h_dim,
        },
        out_file,
    )
    print(f"Saved {tuple(reps.shape)} representations -> {out_file}")


if __name__ == "__main__":
    main()
