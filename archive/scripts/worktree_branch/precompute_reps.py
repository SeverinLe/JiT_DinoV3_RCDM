"""
scripts/precompute_reps.py

Precompute frozen encoder representations for all training images and save
them to disk. Training reads from this file — the encoder never runs during
the training loop.

Origin / changes vs upstream repos
------------------------------------
  FROM RCDM (adapted). RCDM's train.py ran the encoder on every training step
  (online encoding). We split this into a separate precomputation step.

  Changes made:
    1. Encoder: torchvision ResNet-50 (avgpool, 2048-dim)
                → DinoV3 ViT-S/16 (CLS token, 384-dim).

    2. Rep shape: (N, 2048) → (N, 384).

    3. image_size for encoder: was args.image_size (tied to generator resolution).
       Now hardcoded to 224 — DinoV3 ViT-S/16 has a fixed 14×14 positional
       embedding grid (224 / 16 = 14 patches). Using any other resolution would
       silently corrupt positional embeddings. The generator's image_size is a
       separate concern and does not affect the encoder.

    4. data_dir default: data/tiny-imagenet-200/train → data/messidor2/train.

    5. Output file stores {"paths": [...], "reps": Tensor(N, 384)}.
       Index alignment is exact: reps[i] corresponds to paths[i].
       RepresentationDataset relies on this alignment.

  use_gated_mlp prerequisite:
    Before running this script, verify that config.json in the DinoV3 checkpoint
    directory has "use_gated_mlp": false. If it says true, the encoder will
    randomly initialise a missing gate_proj on every load → non-deterministic
    representations. Delete any existing train_reps.pt and rerun after the fix.

Usage:
    python scripts/precompute_reps.py \\
        --data_dir  data/messidor2/train \\
        --out_file  data/messidor2/train_reps.pt \\
        --batch_size 64 \\
        --device cpu
"""

import argparse
import sys
import os
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rcdm.encoder import load_encoder, build_transform, encode_batch


def collect_image_paths(data_dir):
    """
    Walk the training directory and collect every image path.

    Returns a sorted list of absolute path strings.
    Sorted so the order is deterministic across runs — index alignment between
    paths and reps must be preserved exactly.
    """
    # ── changed from RCDM: added .tif/.tiff for Messidor-2 format ──
    valid_extensions = {".jpeg", ".jpg", ".png", ".JPEG", ".tif", ".tiff"}
    paths = []

    data_dir = Path(data_dir)
    for img_path in sorted(data_dir.rglob("*")):
        if img_path.suffix in valid_extensions:
            paths.append(str(img_path))

    print(f"Found {len(paths)} images in {data_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Precompute SSL representations")
    # ── changed from RCDM: default path updated for Messidor-2 ──
    parser.add_argument("--data_dir",   type=str, default="data/messidor2/train")
    parser.add_argument("--out_file",   type=str, default="data/messidor2/train_reps.pt")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Controls the GENERATOR output resolution only. "
                             "The encoder always runs at 224 px regardless of this value.")
    parser.add_argument("--batch_size", type=int, default=128)
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
    print(f"\n[2/3] Loading encoder on {args.device}...")
    encoder = load_encoder(device=args.device)

    # ── JiT-RCDM [fix-2]: encoder transform hardcoded to 224, not args.image_size ──
    # RCDM used: transform = build_transform(image_size=args.image_size)
    # Changed to: always 224 — DinoV3 ViT-S/16 has a fixed 14×14 pos-embed grid.
    # args.image_size controls generator resolution; encoder resolution is independent.
    transform = build_transform(image_size=224)

    print(f"Running encoder over {len(paths)} images (batch_size={args.batch_size})...")
    reps = encode_batch(
        image_paths=paths,
        encoder=encoder,
        transform=transform,
        device=args.device,
        batch_size=args.batch_size,
    )

    # ── changed from RCDM: reps shape is (N, 384) not (N, 2048) ──
    print(f"\nRepresentations shape : {reps.shape}")
    print(f"Representations dtype : {reps.dtype}")
    print(f"Sample norm (first 5) : {reps[:5].norm(dim=1).tolist()}")

    # ── Step 3: save to disk ──────────────────────────────────────────────── #
    print(f"\n[3/3] Saving to {args.out_file}...")
    os.makedirs(Path(args.out_file).parent, exist_ok=True)

    torch.save(
        {
            "paths": paths,   # list[str], length N — absolute paths on this machine
            "reps":  reps,    # Tensor (N, 384) — DinoV3 CLS tokens
        },
        args.out_file,
    )

    # Sanity check: index alignment must be exact
    loaded = torch.load(args.out_file)
    assert len(loaded["paths"]) == loaded["reps"].shape[0], \
        "Path count and rep count don't match — something went wrong"
    # ── changed from RCDM: assert 384-dim, not 2048-dim ──
    assert loaded["reps"].shape[1] == 384, \
        f"Expected 384-dim DinoV3 ViT-S reps, got {loaded['reps'].shape[1]}"

    print(f"\nDone. Saved {len(paths)} representations to {args.out_file}")
    print(f"File size: {Path(args.out_file).stat().st_size / 1e6:.1f} MB")
    print("\nVerification passed — paths and reps are aligned.")


if __name__ == "__main__":
    main()
