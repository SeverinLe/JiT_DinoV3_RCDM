"""
experiments/sample_grid.py

Qualitative sample grids: [conditioning image | k generated samples] per input.

This is the workhorse figure generator, not a probe — it produces Figure 2 of the
report (what the generator makes from a given h) and is the visual companion to
every quantitative experiment.  Encoder-agnostic: pass --encoder and the script
checks it against the checkpoint's recorded h_dim.

Usage:
    python experiments/sample_grid.py \
        --checkpoint models/jit_dinov3/final.pt \
        --encoder    dinov3 \
        --cond_images data/raw/messidor2/test/anodr/*.png \
        --n_samples  4 --cfg_scale 3.0 --num_steps 50 --seed 0 --device mps

Output: experiments/results/sample_grid/<encoder>/<tag>/figures/*.png
"""

import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms

from common import (
    create_run_dir,
    load_generator,
    load_probe_encoder,
    resolve_device,
    set_seed,
    sha256_file,
    unnorm,
)

EXPERIMENT = "sample_grid"


def build_display_transform(image_size: int) -> transforms.Compose:
    """
    Preprocess the conditioning image for *display* in the grid.

    Note this is the [-1, 1] diffusion normalisation at the generator's
    resolution, not the ImageNet normalisation at 224 px used for encoding.
    The two pipelines are deliberately separate.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative sample grids")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--cond_images", nargs="+", required=True)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    model, flow, cfg = load_generator(args.checkpoint, device)
    encoder, enc_transform = load_probe_encoder(args.encoder, device, cfg, args.encoder_ckpt)
    display_transform = build_display_transform(cfg["image_size"])

    run_dir = create_run_dir(
        EXPERIMENT, args.encoder, args.tag, args,
        extra={"checkpoint_sha256": sha256_file(args.checkpoint), "model_cfg": cfg},
    )

    for image_path in args.cond_images:
        image_path = Path(image_path)
        img = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            h = encoder(enc_transform(img).unsqueeze(0).to(device))       # (1, h_dim)
            h_batch = h.repeat(args.n_samples, 1)                          # (k, h_dim)
            noise = torch.randn(
                args.n_samples, 3, cfg["image_size"], cfg["image_size"], device=device
            )
            generated = flow.sample(
                model, noise, h_batch,
                num_steps=args.num_steps, cfg_scale=args.cfg_scale,
            )

        # Row: conditioning image first, then the k samples.
        conditioning = display_transform(img).unsqueeze(0).to(device)
        row = torch.cat([unnorm(conditioning), unnorm(generated)], dim=0)

        out_path = run_dir / "figures" / f"grid_{image_path.stem}.png"
        vutils.save_image(row, out_path, nrow=args.n_samples + 1, padding=2)
        print(f"  {out_path.name}")

    print(f"\nDone — {len(args.cond_images)} grids in {run_dir}")


if __name__ == "__main__":
    main()
