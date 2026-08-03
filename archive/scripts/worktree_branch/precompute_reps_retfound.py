"""
scripts/precompute_reps_retfound.py

Precompute frozen RETFound encoder representations for all training images
and save them to disk. Training reads from this file — the encoder never
runs during the training loop.

Origin / changes vs scripts/precompute_reps.py (DinoV3 ViT-S/16)
------------------------------------------------------------------
  Copy of precompute_reps.py with only the encoder swapped:

    1. Encoder: DinoV3 ViT-S/16 (CLS token, 384-dim)
                → RETFound MAE ViT-Large/16 (CLS token, 1024-dim).

    2. Rep shape: (N, 384) → (N, 1024).

    3. Output file: data/messidor2/train_reps.pt → data/messidor2/train_reps_retfound.pt
       Same format: {"paths": [...], "reps": Tensor(N, 1024)}.
       Index alignment is exact: reps[i] corresponds to paths[i].
       RepresentationDataset relies on this alignment.

    Everything else (image collection, batching, image_size note) is
    unchanged — the encoder always runs at 224x224 regardless of
    --image_size, since RETFound ViT-L/16 also has a fixed 14x14 pos-embed
    grid (224 / 16 = 14 patches).

Usage:
    python scripts/precompute_reps_retfound.py \\
        --data_dir  data/messidor2/train \\
        --out_file  data/messidor2/train_reps_retfound.pt \\
        --batch_size 32 \\
        --device cpu
"""

import argparse
import sys
import os
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rcdm.encoder_retfound import load_encoder, build_transform, encode_batch


def collect_image_paths(data_dir):
    """
    Walk the training directory and collect every image path.

    Returns a sorted list of absolute path strings.
    Sorted so the order is deterministic across runs — index alignment between
    paths and reps must be preserved exactly.
    """
    valid_extensions = {".jpeg", ".jpg", ".png", ".JPEG", ".tif", ".tiff"}
    paths = []

    data_dir = Path(data_dir)
    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix in valid_extensions:
            paths.append(str(img_path))

    print(f"Found {len(paths)} images in {data_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Precompute RETFound representations")
    parser.add_argument("--data_dir",   type=str, default="data/messidor2/train")
    parser.add_argument("--out_file",   type=str, default="data/messidor2/train_reps_retfound.pt")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Controls the GENERATOR output resolution only. "
                             "The encoder always runs at 224 px regardless of this value.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="RETFound ViT-L/16 is much larger than DinoV3 ViT-S/16 — "
                             "reduce this if you run out of memory.")
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    # ── Step 1: collect all image paths ──────────────────────────────────── #
    print("\n[1/3] Collecting image paths...")
    paths = collect_image_paths(args.data_dir)

    if len(paths) == 0:
        raise RuntimeError(
            f"No images found in {args.data_dir}. "
            "Check that the path points to the Messidor-2 train/ folder."
        )

    # ── Step 2: load encoder and run over all images ──────────────────────── #
    print(f"\n[2/3] Loading RETFound encoder on {args.device}...")
    encoder = load_encoder(device=args.device)

    # Encoder transform hardcoded to 224, not args.image_size — RETFound
    # ViT-L/16 has a fixed 14x14 pos-embed grid (same constraint as DinoV3).
    transform = build_transform(image_size=224)

    print(f"Running encoder over {len(paths)} images (batch_size={args.batch_size})...")
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

    # ── Step 3: save to disk ──────────────────────────────────────────────── #
    print(f"\n[3/3] Saving to {args.out_file}...")
    os.makedirs(Path(args.out_file).parent, exist_ok=True)

    torch.save(
        {
            "paths": paths,   # list[str], length N — absolute paths on this machine
            "reps":  reps,    # Tensor (N, 1024) — RETFound ViT-L/16 CLS tokens
        },
        args.out_file,
    )

    # Sanity check: index alignment must be exact
    loaded = torch.load(args.out_file)
    assert len(loaded["paths"]) == loaded["reps"].shape[0], \
        "Path count and rep count don't match — something went wrong"
    assert loaded["reps"].shape[1] == 1024, \
        f"Expected 1024-dim RETFound ViT-L/16 reps, got {loaded['reps'].shape[1]}"

    print(f"\nDone. Saved {len(paths)} representations to {args.out_file}")
    print(f"File size: {Path(args.out_file).stat().st_size / 1e6:.1f} MB")
    print("\nVerification passed — paths and reps are aligned.")


if __name__ == "__main__":
    main()
