"""
scripts/invariance_images.py

Build an invariance test set for probing the DinoV3 backbone and the JiT-RCDM
generator.

Picks ONE image from each of the five DR-grade folders in data/MESSIDOR2/test
(anodr, bmilddr, cmoderatedr, dseveredr, eproliferativedr) and writes, for each
picked image, the original plus five deterministic augmentations into
test_images_invariance/<grade>/:

    original.png          ← unmodified source image
    crop128.png           ← (1) centre crop to 128x128
    rot90.png             ← (2) rotation 90 degrees to the right (clockwise)
    mirror_vertical.png   ← (3) vertical mirror (flip top<->bottom)
    translate.png         ← (4) random perturbation / translation (seeded)
    grayscale.png         ← (5) grayscale (kept as 3 identical channels)

The augmentations are applied to the full-resolution source image. The encoder
(DinoV3) and the generator both resize to their own fixed resolution at
inference time, so the augmentations only need to be semantically meaningful,
not pre-sized.

Usage (from the project root or the worktree root):
    python scripts/invariance_images.py \\
        --test_dir data/MESSIDOR2/test \\
        --out_dir  test_images_invariance \\
        --seed     0
"""

import argparse
import json
from pathlib import Path

from PIL import Image
import torchvision.transforms.functional as TF
import torch


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def pick_image(folder: Path, mode: str, rng: torch.Generator):
    imgs = list_images(folder)
    if not imgs:
        return None
    if mode == "random":
        idx = int(torch.randint(len(imgs), (1,), generator=rng).item())
        return imgs[idx]
    return imgs[0]  # deterministic: first image when sorted


def make_augmentations(img: Image.Image, crop_size: int, rng: torch.Generator):
    """Return an ordered dict {name: PIL.Image} of original + 5 augmentations."""
    out = {}

    # (0) original, untouched
    out["original"] = img.copy()

    # (1) centre crop to crop_size x crop_size
    out["crop128"] = TF.center_crop(img, [crop_size, crop_size])

    # (1b) zoom x2: centre crop to half the image (1384 -> 692), a true 2x zoom.
    #      build_transform(224) downsamples this back to 224, so detail survives.
    w0, h0 = img.size
    out["zoom_x2"] = TF.center_crop(img, [h0 // 2, w0 // 2])

    # (2) rotation 90 degrees to the right == clockwise.
    #     TF.rotate uses counter-clockwise angles, so clockwise 90 == -90.
    out["rot90"] = TF.rotate(img, angle=-90, expand=True)

    # (3) mirror vertically == flip top<->bottom.
    out["mirror_vertical"] = TF.vflip(img)

    # (3b) mirror horizontally == flip left<->right.
    out["mirror_horizontal"] = TF.hflip(img)

    # (4) random perturbation / translation: shift by up to +/-20% of each side,
    #     black fill for the revealed border. Seeded for reproducibility.
    w, h = img.size
    max_dx, max_dy = int(0.20 * w), int(0.20 * h)
    dx = int(torch.randint(-max_dx, max_dx + 1, (1,), generator=rng).item())
    dy = int(torch.randint(-max_dy, max_dy + 1, (1,), generator=rng).item())
    out["translate"] = TF.affine(
        img, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0], fill=0
    )

    # (5) grayscale, kept as 3 identical channels so the RGB encoder still works.
    out["grayscale"] = TF.to_grayscale(img, num_output_channels=3)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="data/MESSIDOR2/test",
                        help="Directory containing the five DR-grade subfolders")
    parser.add_argument("--out_dir", type=str, default="test_images_invariance",
                        help="Where to write the augmented invariance set")
    parser.add_argument("--crop_size", type=int, default=128,
                        help="Side length for the centre-crop augmentation")
    parser.add_argument("--pick", type=str, default="first",
                        choices=["first", "random"],
                        help="Which image to take from each folder")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for image pick + random translation")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = torch.Generator().manual_seed(args.seed)

    subfolders = sorted(p for p in test_dir.iterdir() if p.is_dir())
    if not subfolders:
        raise SystemExit(f"No grade subfolders found in {test_dir}")

    manifest = {}
    print(f"Building invariance set from {test_dir} -> {out_dir}\n")

    for folder in subfolders:
        grade = folder.name
        src = pick_image(folder, args.pick, rng)
        if src is None:
            print(f"  [{grade}] no images found, skipping")
            continue

        img = Image.open(src).convert("RGB")
        variants = make_augmentations(img, args.crop_size, rng)

        grade_out = out_dir / grade
        grade_out.mkdir(parents=True, exist_ok=True)

        saved = {}
        for name, v in variants.items():
            path = grade_out / f"{name}.png"
            v.save(path)
            saved[name] = str(path)

        manifest[grade] = {"source": str(src), "variants": saved}
        print(f"  [{grade}] source={src.name}  ->  {len(variants)} variants")

    manifest_path = out_dir / "manifest.json"
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nDone. {len(manifest)} grades written to {out_dir}/")
        print(f"Manifest: {manifest_path}")
    except OSError as e:
        print(f"\nDone. {len(manifest)} grades written to {out_dir}/")
        print(f"(could not write manifest.json: {e})")


if __name__ == "__main__":
    main()
