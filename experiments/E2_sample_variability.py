"""
experiments/E2_sample_variability.py — RQ2: what does h encode, and what does it discard?

The RCDM reading of a conditional generative model: sample k images from one
conditioning vector h and look at what stays constant versus what varies.
Structure that is *stable* across samples is encoded in h; structure that
*varies* is information the encoder discarded and the generator is free to
invent.

Where the earlier grid-only version of this analysis stopped at "the samples
differ", this script quantifies the difference, which is what lets the result
carry an error bar:

  per_pixel_std     mean over pixels of the std across the k samples
  ssim_between      mean pairwise SSIM among the k samples (high = consistent)
  ssim_to_real      mean SSIM of each sample against the conditioning image
  lpips_between     optional perceptual distance, if lpips is installed

Reading the two SSIMs together separates the two failure modes: low
ssim_between means the representation underdetermines the image; high
ssim_between with low ssim_to_real means h pins down *an* image consistently,
just not the right one.

Figures per conditioning image:
  samples_<stem>.png       [real | sample_1 ... sample_k]
  variability_<stem>.png   mean sample, per-pixel std map, |mean - real|

Usage:
    python experiments/E2_sample_variability.py \
        --checkpoint models/jit_dinov3/final.pt \
        --encoder    dinov3 \
        --test_dir   data/raw/messidor2/test \
        --n_per_class 6 --n_samples 8 --seed 0 --device mps
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms

from common import (
    create_run_dir,
    load_generator,
    load_probe_encoder,
    resolve_device,
    save_metrics,
    set_plot_style,
    set_seed,
    sha256_file,
    stratified_sample,
    unnorm,
)

EXPERIMENT = "E2_sample_variability"


def pairwise_ssim(images: np.ndarray) -> float:
    """
    Mean SSIM over all unordered pairs of a (k, H, W, 3) stack in [0, 1].

    Returns NaN if scikit-image is unavailable, so the rest of the metrics
    still get written.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")
    scores = [
        structural_similarity(images[i], images[j], channel_axis=2, data_range=1.0)
        for i, j in combinations(range(len(images)), 2)
    ]
    return float(np.mean(scores)) if scores else float("nan")


def ssim_against(reference: np.ndarray, images: np.ndarray) -> float:
    """Mean SSIM of each image in the stack against a single reference."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")
    return float(np.mean([
        structural_similarity(reference, img, channel_axis=2, data_range=1.0)
        for img in images
    ]))


def to_numpy_stack(x: torch.Tensor) -> np.ndarray:
    """(k, 3, H, W) in [0, 1] -> (k, H, W, 3) float numpy."""
    return x.permute(0, 2, 3, 1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 — what h encodes vs. discards")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--test_dir", default="data/raw/messidor2/test")
    parser.add_argument("--n_per_class", type=int, default=6,
                        help="Conditioning images sampled per class")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Generations per conditioning image (k). The "
                             "variability estimate is noisy below ~8.")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    set_plot_style()

    model, flow, cfg = load_generator(args.checkpoint, device)
    encoder, enc_transform = load_probe_encoder(args.encoder, device, cfg, args.encoder_ckpt)
    image_size = cfg["image_size"]

    display_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    pairs = stratified_sample(args.test_dir, args.n_per_class, seed=args.seed)
    print(f"  [data] {len(pairs)} conditioning images x {args.n_samples} samples")

    run_dir = create_run_dir(
        EXPERIMENT, args.encoder, args.tag, args,
        extra={"checkpoint_sha256": sha256_file(args.checkpoint), "model_cfg": cfg,
               "n_conditioning_images": len(pairs)},
    )

    import matplotlib.pyplot as plt

    rows = []
    for class_name, path in pairs:
        img = Image.open(path).convert("RGB")

        with torch.no_grad():
            h = encoder(enc_transform(img).unsqueeze(0).to(device))
            noise = torch.randn(args.n_samples, 3, image_size, image_size, device=device)
            generated = flow.sample(model, noise, h.repeat(args.n_samples, 1),
                                    num_steps=args.num_steps, cfg_scale=args.cfg_scale)

        samples = unnorm(generated)                      # (k, 3, H, W) in [0, 1]
        real = display_transform(img).to(device)         # (3, H, W) in [0, 1]

        stack = to_numpy_stack(samples)
        real_np = real.permute(1, 2, 0).cpu().numpy()

        std_map = samples.std(dim=0).mean(dim=0)         # (H, W) across samples
        metrics = {
            "image": path.stem,
            "class": class_name,
            "per_pixel_std": float(std_map.mean()),
            "per_pixel_std_max": float(std_map.max()),
            "ssim_between": pairwise_ssim(stack),
            "ssim_to_real": ssim_against(real_np, stack),
        }
        rows.append(metrics)

        # Figure 1: the grid itself.
        vutils.save_image(
            torch.cat([real.unsqueeze(0), samples], dim=0),
            run_dir / "figures" / f"samples_{class_name}_{path.stem}.png",
            nrow=args.n_samples + 1, padding=2,
        )

        # Figure 2: where the samples agree and where they do not.
        mean_sample = samples.mean(dim=0)
        abs_diff = (mean_sample - real).abs().mean(dim=0)
        fig, axes = plt.subplots(1, 4, figsize=(11, 3))
        for ax, (title, data, cmap) in zip(axes, [
            ("conditioning image", real_np, None),
            (f"mean of k={args.n_samples}", mean_sample.permute(1, 2, 0).cpu().numpy(), None),
            ("per-pixel std (discarded)", std_map.cpu().numpy(), "magma"),
            ("|mean - real|", abs_diff.cpu().numpy(), "magma"),
        ]):
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            if cmap:
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{class_name} / {path.stem} — "
                     f"SSIM between samples {metrics['ssim_between']:.3f}, "
                     f"to real {metrics['ssim_to_real']:.3f}", fontsize=9)
        fig.savefig(run_dir / "figures" / f"variability_{class_name}_{path.stem}.png")
        plt.close(fig)

        print(f"  {class_name}/{path.stem}: std {metrics['per_pixel_std']:.4f}, "
              f"ssim_between {metrics['ssim_between']:.3f}")

    save_metrics(run_dir, "metrics", rows,
                 ["image", "class", "per_pixel_std", "per_pixel_std_max",
                  "ssim_between", "ssim_to_real"])

    # Aggregate per class — advanced grades have few training images, so a
    # per-class breakdown is where the data-scarcity effect shows up.
    summary = {}
    for key in ("per_pixel_std", "ssim_between", "ssim_to_real"):
        values = np.array([r[key] for r in rows], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": int(np.count_nonzero(~np.isnan(values))),
        }
    summary["per_class"] = {
        class_name: {
            key: float(np.nanmean([r[key] for r in rows if r["class"] == class_name]))
            for key in ("per_pixel_std", "ssim_between", "ssim_to_real")
        }
        for class_name in sorted({r["class"] for r in rows})
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  mean per-pixel std {summary['per_pixel_std']['mean']:.4f} "
          f"± {summary['per_pixel_std']['std']:.4f}")
    print(f"Done — {run_dir}")


if __name__ == "__main__":
    main()
