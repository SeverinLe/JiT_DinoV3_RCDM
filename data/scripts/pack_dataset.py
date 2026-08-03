"""
scripts/pack_dataset.py

Pack a standard train_reps.pt (paths + reps) into a self-contained file
that also embeds the image tensors as uint8.

The packed file can be uploaded to Google Drive and used on Colab without
needing to transfer the raw image folder.

Packed format:
    {
        "images": Tensor(N, 3, H, W)  uint8  [0-255]   ~150 MB for 972 × 224px
        "reps":   Tensor(N, 384)      float32           ~1.4 MB
    }

RepresentationDataset automatically detects the packed format and converts
uint8 → float32 → [-1, 1] on the fly (fast, no disk I/O at training time).

Usage:
    python scripts/pack_dataset.py \
        --reps_file  data/messidor2/train_reps.pt \
        --out_file   data/messidor2/train_packed.pt \
        --image_size 224
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps_file",  type=str,
                        default="data/messidor2/train_reps.pt")
    parser.add_argument("--out_file",   type=str,
                        default="data/messidor2/train_packed.pt")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    print(f"Loading {args.reps_file}...")
    data  = torch.load(args.reps_file, weights_only=False)
    paths = data["paths"]
    reps  = data["reps"]
    N     = len(paths)
    S     = args.image_size
    print(f"  {N} images, reps shape {tuple(reps.shape)}")

    images = torch.zeros(N, 3, S, S, dtype=torch.uint8)

    print(f"Encoding {N} images at {S}×{S} → uint8 ...")
    for i, p in enumerate(tqdm(paths)):
        img = Image.open(p).convert("RGB")
        img = TF.resize(img, S, interpolation=TF.InterpolationMode.BICUBIC)
        img = TF.center_crop(img, S)
        t   = TF.to_tensor(img)              # float32 [0, 1]
        images[i] = (t * 255).clamp(0, 255).byte()

    out = {"images": images, "reps": reps}
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_file)

    size_mb = Path(args.out_file).stat().st_size / 1e6
    print(f"\n✓ Saved packed dataset → {args.out_file}  ({size_mb:.0f} MB)")
    print(f"  images : {tuple(images.shape)}  dtype={images.dtype}")
    print(f"  reps   : {tuple(reps.shape)}    dtype={reps.dtype}")
    print("\nUpload this file to Google Drive at MyDrive/jit_rcdm/train_packed.pt")


if __name__ == "__main__":
    main()
