"""
experiments/E2_variance_heatmap.py — RQ2, the visual companion to E2.

E2_sample_variability.py answers "how much does h leave undetermined" with a
number per conditioning image.  This script answers "*where*" with a picture:
for each of a handful of conditioning images it generates k samples, takes the
per-pixel standard deviation across them, and renders that map over the retina.

Bright regions are pixels the generator felt free to change while still
satisfying the same h — candidate discarded information.  Dark regions are
pixels h pinned down.

Two things make the figure readable that a per-image plot does not give you:

  1. **One shared colour scale across every row.**  Variance maps normalised
     per-panel look identical for a well-determined and a badly-determined
     image, because each is stretched to its own range.  A single vmin/vmax over
     the whole figure is what lets you say "grade 4 varies more than grade 0".
  2. **A retina mask.**  After Resize + CenterCrop the frame still has dark
     corners outside the fundus disc, where every sample is black and the std is
     ~0.  Averaging those in drags every summary number toward zero by a
     constant factor.  The reported means are taken inside the mask; the maps
     show the mask boundary so it can be checked by eye.

Default is one conditioning image per DR grade (5 rows), which is the
comparison the report wants.  ``--n_per_class`` raises that.

Output:
  figures/variance_heatmap_grid.png    all rows, shared colour scale
  figures/heatmap_<class>_<stem>.png   one panel per conditioning image
  metrics.csv                          per-image masked std, in and out of mask
  summary.json                         per-class aggregates + the shared scale

Usage:
    python experiments/E2_variance_heatmap.py \
        --checkpoint models/jit_dinov3/final.pt \
        --encoder    dinov3 \
        --n_per_class 1 --n_samples 8 --seed 0 --device mps
"""

import argparse
import json

import numpy as np
import torch
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

EXPERIMENT = "E2_variance_heatmap"

# 8-bit quantisation step; below this the k samples are byte-identical and the
# heatmap renders floating-point noise, not discarded information.
QUANTISATION_FLOOR = 1.0 / 255.0

# Fundus photographs are a bright disc on a black surround.  Luminance is a
# sufficient separator; the threshold is deliberately low so that dark pathology
# inside the disc is not excluded along with the background.
RETINA_LUMA_THRESHOLD = 0.08


def retina_mask(real: torch.Tensor, threshold: float = RETINA_LUMA_THRESHOLD) -> torch.Tensor:
    """
    Boolean (H, W) mask of the fundus disc in a (3, H, W) image in [0, 1].

    Uses the conditioning image rather than the generations: the mask should be
    a property of the scan being probed, not of what the generator happened to
    produce.
    """
    return real.mean(dim=0) > threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="E2 — where does the sample variance live")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--test_dir", default="data/raw/messidor2/test")
    parser.add_argument("--n_per_class", type=int, default=1,
                        help="Conditioning images per class. 1 gives the "
                             "5-row one-per-grade figure the report uses.")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Generations per conditioning image (k). The "
                             "per-pixel std is noisy below ~8.")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="1.0 = no guidance, and the default for every probe. "
                             "CFG > 1 suppresses exactly the variance this maps.")
    parser.add_argument("--mask_background", action="store_true", default=True,
                        help="Report std inside the fundus disc only (default).")
    parser.add_argument("--no_mask_background", dest="mask_background",
                        action="store_false")
    parser.add_argument("--cmap", default="magma")
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

    # Bicubic to match data/scripts/pack_dataset.py, i.e. the pixels the
    # generator was actually trained on.
    display_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
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

    # ---- generate and collect the maps before plotting ----------------------
    # The shared colour scale needs every map in hand first, so this is a
    # two-pass script: compute all, then render.
    panels, rows = [], []
    for class_name, path in pairs:
        img = Image.open(path).convert("RGB")

        with torch.no_grad():
            h = encoder(enc_transform(img).unsqueeze(0).to(device))
            noise = torch.randn(args.n_samples, 3, image_size, image_size, device=device)
            generated = flow.sample(model, noise, h.repeat(args.n_samples, 1),
                                    num_steps=args.num_steps, cfg_scale=args.cfg_scale)

        samples = unnorm(generated).cpu()                 # (k, 3, H, W) in [0, 1]
        real = display_transform(img)                     # (3, H, W) in [0, 1]

        std_map = samples.std(dim=0).mean(dim=0)          # (H, W) across samples
        mask = retina_mask(real)
        inside = std_map[mask]
        outside = std_map[~mask]

        panels.append({
            "class": class_name, "stem": path.stem,
            "real": real.permute(1, 2, 0).numpy(),
            "mean": samples.mean(dim=0).permute(1, 2, 0).numpy(),
            "std": std_map.numpy(), "mask": mask.numpy(),
        })
        rows.append({
            "image": path.stem, "class": class_name,
            "std_in_retina": float(inside.mean()) if inside.numel() else float("nan"),
            "std_outside_retina": float(outside.mean()) if outside.numel() else float("nan"),
            "std_whole_frame": float(std_map.mean()),
            "std_max": float(std_map.max()),
            "retina_fraction": float(mask.float().mean()),
        })
        print(f"  {class_name}/{path.stem}: std in-retina "
              f"{rows[-1]['std_in_retina']:.4f} (whole frame "
              f"{rows[-1]['std_whole_frame']:.4f})")

    save_metrics(run_dir, "metrics", rows,
                 ["image", "class", "std_in_retina", "std_outside_retina",
                  "std_whole_frame", "std_max", "retina_fraction"])

    # ---- collapse guard -----------------------------------------------------
    # Everything below normalises to the observed range, so a generator that
    # ignores its noise still produces a vivid-looking heatmap — of numerical
    # noise, amplified by the colour scale.  Check before rendering.
    peak_std = max(float(np.max(p["std"])) for p in panels)
    collapsed = peak_std < QUANTISATION_FLOOR
    if collapsed:
        print(f"\n  {'=' * 68}")
        print(f"  SAMPLER COLLAPSE: peak per-pixel std {peak_std:.6f} is below the "
              f"8-bit\n  quantisation floor {QUANTISATION_FLOOR:.4f} — the k samples "
              "are byte-identical.")
        print("  The heatmaps below show floating-point noise, not discarded "
              "information.\n  Do not read them as anatomy. See "
              "E2_sample_variability.py for the diagnosis.")
        print(f"  {'=' * 68}\n")

    # ---- shared colour scale ------------------------------------------------
    # Percentile rather than max: a handful of hot border pixels would otherwise
    # compress every map into the bottom of the colour range.
    scored = np.concatenate([
        p["std"][p["mask"]] if args.mask_background else p["std"].ravel()
        for p in panels
    ])
    vmin, vmax = 0.0, float(np.percentile(scored, 99))
    print(f"  [scale] shared colour range 0 - {vmax:.4f} (99th pct)")

    # ---- the grid figure ----------------------------------------------------
    n = len(panels)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.7 * n), squeeze=False)
    for row, panel in zip(axes, panels):
        heat = np.ma.masked_where(~panel["mask"], panel["std"]) \
            if args.mask_background else panel["std"]

        row[0].imshow(panel["real"])
        row[0].set_ylabel(panel["class"], fontsize=8)
        row[0].set_title("conditioning image", fontsize=8)

        row[1].imshow(panel["mean"])
        row[1].set_title(f"mean of k={args.n_samples}", fontsize=8)

        im = row[2].imshow(heat, cmap=args.cmap, vmin=vmin, vmax=vmax)
        row[2].set_title("per-pixel std across samples", fontsize=8)

        # Overlay anchors the variance to anatomy — without it the map is hard
        # to attribute to disc, vessels or periphery.
        row[3].imshow(panel["real"])
        row[3].imshow(heat, cmap=args.cmap, vmin=vmin, vmax=vmax, alpha=0.65)
        row[3].set_title("overlay", fontsize=8)

        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"E2 — where the samples disagree ({args.encoder}, k={args.n_samples}, "
        f"cfg {args.cfg_scale})\nshared colour scale 0-{vmax:.3f}; "
        f"bright = h left it free",
        fontsize=10,
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01,
                 label="std across samples")
    fig.savefig(run_dir / "figures" / "variance_heatmap_grid.png")
    plt.close(fig)

    # ---- one standalone panel per image -------------------------------------
    for panel in panels:
        heat = np.ma.masked_where(~panel["mask"], panel["std"]) \
            if args.mask_background else panel["std"]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(panel["real"])
        im = ax.imshow(heat, cmap=args.cmap, vmin=vmin, vmax=vmax, alpha=0.65)
        ax.set_title(f"{panel['class']} / {panel['stem']}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(run_dir / "figures"
                    / f"heatmap_{panel['class']}_{panel['stem']}.png")
        plt.close(fig)

    # ---- summary ------------------------------------------------------------
    summary = {
        "n_images": len(rows), "k": args.n_samples,
        "cfg_scale": args.cfg_scale, "num_steps": args.num_steps, "seed": args.seed,
        "masked": bool(args.mask_background),
        "sampler_collapsed": bool(collapsed),
        "peak_per_pixel_std": peak_std,
        "quantisation_floor": QUANTISATION_FLOOR,
        "shared_scale": {"vmin": vmin, "vmax": vmax, "percentile": 99},
        "std_in_retina": {
            "mean": float(np.nanmean([r["std_in_retina"] for r in rows])),
            "std": float(np.nanstd([r["std_in_retina"] for r in rows], ddof=1))
            if len(rows) > 1 else 0.0,
        },
        "per_class": {
            class_name: {
                key: float(np.nanmean([r[key] for r in rows if r["class"] == class_name]))
                for key in ("std_in_retina", "std_outside_retina", "std_whole_frame")
            }
            for class_name in sorted({r["class"] for r in rows})
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  mean std inside retina {summary['std_in_retina']['mean']:.4f}")
    print("  Absolute values need E2_sample_variability's between-h baseline to "
          "be read;\n  this script is for localisation, not magnitude.")
    print(f"Done — {run_dir}")


if __name__ == "__main__":
    main()
