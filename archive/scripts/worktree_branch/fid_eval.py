"""
scripts/fid_eval.py

Pick N images from each subdirectory of a test folder, generate 1 sample per
image using a trained JiT-RCDM checkpoint, and compute FID between the real
and generated sets.

Outputs
-------
  <out_dir>/fid_singles/   generated images only  (sample_<class>_<stem>.png)
  <out_dir>/fid_real/      corresponding real images (real_<class>_<stem>.png)

Usage
-----
    python scripts/fid_eval.py \\
        --checkpoint checkpoints/jit_rcdm_step0100000.pt \\
        --test_dir   /path/to/MESSIDOR2/test/ \\
        --out_dir    samples/ \\
        --n_per_class 2 \\
        --cfg_scale  1.0 \\
        --num_steps  50 \\
        --device     mps
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ── make sure rcdm/ is importable regardless of CWD ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcdm.encoder import load_encoder, build_transform
from rcdm.jit import create_jit_model, FlowMatching


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG"}


def pick_images(test_dir: Path, n: int) -> list[tuple[str, Path]]:
    """
    Return a list of (class_name, image_path) tuples.
    Takes the first N images (sorted) from each subdirectory.
    """
    pairs = []
    subdirs = sorted(p for p in test_dir.iterdir() if p.is_dir())
    if not subdirs:
        raise RuntimeError(
            f"No subdirectories found in {test_dir}. "
            "Expected structure: test/<class>/<image>.png"
        )
    for subdir in subdirs:
        imgs = sorted(p for p in subdir.iterdir() if p.suffix in IMG_EXTS)
        if len(imgs) == 0:
            print(f"  [warn] {subdir.name}: no images found, skipping")
            continue
        chosen = imgs[:n]
        for img_path in chosen:
            pairs.append((subdir.name, img_path))
        print(f"  {subdir.name}: picked {len(chosen)}/{len(imgs)} images")
    return pairs


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert (3, H, W) float32 in [-1, 1] → PIL RGB image."""
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JiT-RCDM FID evaluation")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--test_dir",    type=str, required=True,
                        help="Root of test directory with class subdirectories")
    parser.add_argument("--out_dir",     type=str, default="samples/",
                        help="Parent output directory (fid_singles/ and fid_real/ created inside)")
    parser.add_argument("--n_per_class", type=int, default=2,
                        help="Number of images to sample from each class directory")
    parser.add_argument("--cfg_scale",   type=float, default=1.0,
                        help="CFG guidance scale (use 1.0 until 15k steps, then increase)")
    parser.add_argument("--num_steps",   type=int, default=50,
                        help="Heun ODE steps")
    parser.add_argument("--encoder_ckpt", type=str,
                        default="checkpoints/dinov3_vits16_tmp",
                        help="Path to DinoV3 HuggingFace checkpoint directory")
    parser.add_argument("--device",      type=str, default="cpu")
    args = parser.parse_args()

    device    = torch.device(args.device)
    test_dir  = Path(args.test_dir)
    out_dir   = Path(args.out_dir)
    gen_dir   = out_dir / "fid_singles"
    real_dir  = out_dir / "fid_real"
    gen_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Pick images ────────────────────────────────────────────────────────
    print(f"\n[1/4] Picking {args.n_per_class} images per class from {test_dir}")
    pairs = pick_images(test_dir, args.n_per_class)
    print(f"  Total: {len(pairs)} conditioning images")

    # ── 2. Load encoder ──────────────────────────────────────────────────────
    print(f"\n[2/4] Loading encoder from {args.encoder_ckpt} ...")
    encoder   = load_encoder(device=device, checkpoint_path=args.encoder_ckpt)
    transform = build_transform(image_size=224)
    print("  Encoder ready.")

    # ── 3. Load JiT model ────────────────────────────────────────────────────
    print(f"\n[3/4] Loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt["model_cfg"]
    model = create_jit_model(**cfg)

    if "ema" in ckpt:
        # EMA state dict is {"decay": float, "shadow": {param_name: tensor, ...}}
        ema_weights = ckpt["ema"]["shadow"]
        # strict=False: freqs_cis is a register_buffer computed in __init__, not tracked by EMA
        model.load_state_dict(ema_weights, strict=False)
        print(f"  Loaded EMA shadow weights  (step {ckpt.get('step', '?')})")
    else:
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  Loaded raw weights  (no EMA in checkpoint)")

    model.eval().to(device)
    flow = FlowMatching()

    image_size = cfg.get("image_size", 224)
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params  "
          f"image_size={image_size}  cfg_scale={args.cfg_scale}")

    # ── 4. Generate + save ───────────────────────────────────────────────────
    print(f"\n[4/4] Generating {len(pairs)} images ...")
    for i, (class_name, img_path) in enumerate(pairs):
        stem     = img_path.stem
        gen_name  = f"sample_{class_name}_{stem}.png"
        real_name = f"real_{class_name}_{stem}.png"

        # Encode conditioning image
        pil_img = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            pv  = transform(pil_img).unsqueeze(0).to(device)
            out = encoder(pixel_values=pv)
            h   = out.last_hidden_state[:, 0, :]    # (1, 384) CLS token

            # Generate one sample
            noise  = torch.randn(1, 3, image_size, image_size, device=device)
            sample = flow.sample(model, noise, h=h,
                                 num_steps=args.num_steps,
                                 cfg_scale=args.cfg_scale)   # (1, 3, H, W) in [-1,1]

        # Save generated image
        tensor_to_pil(sample[0]).save(gen_dir / gen_name)

        # Save real image (resized to match generator resolution for fair FID)
        real_resized = pil_img.resize((image_size, image_size), Image.BICUBIC)
        real_resized.save(real_dir / real_name)

        print(f"  [{i+1:2d}/{len(pairs)}]  {class_name}/{stem}  →  {gen_name}")

    # ── Summary + FID command ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Generated images : {gen_dir}  ({len(list(gen_dir.glob('*.png')))} files)")
    print(f"Real images      : {real_dir}  ({len(list(real_dir.glob('*.png')))} files)")
    print(f"\nRun FID:")
    print(f"  python -m pytorch_fid {real_dir} {gen_dir} --device cpu")
    print(f"{'─'*60}\n")

    # ── Run FID inline ────────────────────────────────────────────────────────
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        print("Computing FID ...")
        fid = calculate_fid_given_paths(
            [str(real_dir), str(gen_dir)],
            batch_size=min(len(pairs), 8),
            device=torch.device("cpu"),   # always cpu for Inception
            dims=2048,
            num_workers=0,
        )
        print(f"\n  FID = {fid:.2f}")
        print(f"\n  Note: FID with {len(pairs)} images has high variance.")
        print(f"  Use this to compare checkpoints (direction), not as absolute quality.")
    except Exception as e:
        print(f"  (FID computation failed: {e})")
        print(f"  Run manually: python -m pytorch_fid {real_dir} {gen_dir} --device cpu")


if __name__ == "__main__":
    main()
