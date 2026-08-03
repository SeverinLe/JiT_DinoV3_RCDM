"""
experiments/E3_invariance.py — RQ3: which transformations is h invariant to?

Merges the previous invariance_images.py + invariance_probe.py into one script
and scales the measurement from one image per class to a stratified sample, so
the cosine similarities become distributions with confidence intervals rather
than anecdotes.

Two levels of measurement
-------------------------
  (a) **Backbone invariance** — cosine similarity between h(T(x)) and h(x) for
      each transform T.  A perfectly invariant encoder returns 1.0.
  (b) **Generator-visible invariance** (optional, --n_visual > 0) — generate
      from h(T(x)) reusing the *same* starting noise across all variants, so any
      visual difference between rows is attributable to h alone rather than to
      the noise draw.  This is the controlled version; drawing fresh noise per
      variant instead mixes the two factors and answers a different question.

Why this matters clinically
---------------------------
Nuisance factors that survive into h are factors a downstream classifier can
silently latch onto.  Greyscale and intensity/contrast shifts stand in for
device and acquisition-protocol differences; if h moves under them, a grading
model trained on one site's images carries a hidden dependence on that site.

Statistics
----------
Per transform: mean cosine ± bootstrap 95% CI over images.  Transforms are
compared against the least-disruptive one with a paired Wilcoxon signed-rank
test (same images under both conditions), Holm-corrected across the family.

Usage:
    python experiments/E3_invariance.py \
        --checkpoint models/jit_dinov3/final.pt \
        --encoder    dinov3 \
        --test_dir   data/raw/messidor2/test \
        --n_per_class 30 --n_visual 1 --seed 0 --device mps

Set --skip_generation to measure the encoder alone (no checkpoint needed).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image

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

EXPERIMENT = "E3_invariance"

# Display order; also the order of rows in the qualitative grids.
VARIANT_ORDER = [
    "original",
    "crop_center",
    "zoom_x2",
    "rot90",
    "mirror_horizontal",
    "mirror_vertical",
    "translate",
    "grayscale",
    "contrast_low",
    "brightness_up",
]


def make_variants(img: Image.Image, crop_size: int, rng: torch.Generator) -> dict:
    """
    Ordered dict {variant_name: PIL image} of the original plus each transform.

    Transforms are applied at the source resolution; the encoder resizes to
    224 px itself, so they only need to be semantically meaningful.

    The last two (contrast, brightness) are the clinically-motivated additions:
    they stand in for acquisition and device variation, which flips and rotations
    do not model.
    """
    width, height = img.size
    variants = {"original": img.copy()}

    variants["crop_center"] = TF.center_crop(img, [crop_size, crop_size])
    # True 2x zoom: crop the central half, which the encoder then upsamples back.
    variants["zoom_x2"] = TF.center_crop(img, [height // 2, width // 2])
    # TF.rotate takes counter-clockwise angles, so clockwise 90 is -90.
    variants["rot90"] = TF.rotate(img, angle=-90, expand=True)
    variants["mirror_horizontal"] = TF.hflip(img)
    variants["mirror_vertical"] = TF.vflip(img)

    # Shift by up to +/-20% per axis, black fill on the revealed border.
    max_dx, max_dy = int(0.20 * width), int(0.20 * height)
    dx = int(torch.randint(-max_dx, max_dx + 1, (1,), generator=rng).item())
    dy = int(torch.randint(-max_dy, max_dy + 1, (1,), generator=rng).item())
    variants["translate"] = TF.affine(img, angle=0.0, translate=[dx, dy],
                                      scale=1.0, shear=[0.0, 0.0], fill=0)

    # Kept as 3 identical channels so the RGB encoder still applies.
    variants["grayscale"] = TF.to_grayscale(img, num_output_channels=3)
    variants["contrast_low"] = TF.adjust_contrast(img, contrast_factor=0.6)
    variants["brightness_up"] = TF.adjust_brightness(img, brightness_factor=1.3)

    return {name: variants[name] for name in VARIANT_ORDER if name in variants}


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    """Percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = [values[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def holm_correction(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down correction over a family of tests."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    corrected, running_max = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        adjusted = min(1.0, (m - i) * p)
        running_max = max(running_max, adjusted)  # enforce monotonicity
        corrected[key] = running_max
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 — representation invariance")
    parser.add_argument("--checkpoint", default=None,
                        help="Required unless --skip_generation")
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--test_dir", default="data/raw/messidor2/test")
    parser.add_argument("--n_per_class", type=int, default=30,
                        help="Images per class for the cosine-similarity statistics")
    parser.add_argument("--n_visual", type=int, default=1,
                        help="Images per class that additionally get a generated "
                             "comparison grid (0 = statistics only)")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Generations per variant in the qualitative grids")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--skip_generation", action="store_true",
                        help="Measure the encoder only; no generator needed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    if not args.skip_generation and not args.checkpoint:
        parser.error("--checkpoint is required unless --skip_generation is set")

    device = resolve_device(args.device)
    set_seed(args.seed)
    set_plot_style()
    rng = torch.Generator().manual_seed(args.seed)

    cfg, model, flow = None, None, None
    if not args.skip_generation:
        model, flow, cfg = load_generator(args.checkpoint, device)
    encoder, enc_transform = load_probe_encoder(args.encoder, device, cfg, args.encoder_ckpt)

    pairs = stratified_sample(args.test_dir, args.n_per_class, seed=args.seed)
    print(f"  [data] {len(pairs)} images over {len(set(c for c, _ in pairs))} classes")

    extra = {"n_images": len(pairs), "variants": VARIANT_ORDER}
    if args.checkpoint:
        extra["checkpoint_sha256"] = sha256_file(args.checkpoint)
        extra["model_cfg"] = cfg
    run_dir = create_run_dir(EXPERIMENT, args.encoder, args.tag, args, extra=extra)

    # ---- (a) backbone invariance -------------------------------------------
    rows, per_variant = [], {name: [] for name in VARIANT_ORDER}

    for class_name, path in pairs:
        img = Image.open(path).convert("RGB")
        variants = make_variants(img, args.crop_size, rng)

        with torch.no_grad():
            x = torch.stack([enc_transform(v) for v in variants.values()]).to(device)
            h = encoder(x)                                   # (V, h_dim)
        h_original = h[0:1]
        cosines = F.cosine_similarity(h, h_original.expand_as(h), dim=1).cpu().numpy()

        for name, cosine in zip(variants, cosines):
            rows.append({"image": path.stem, "class": class_name,
                         "variant": name, "cosine": float(cosine)})
            per_variant[name].append(float(cosine))

    save_metrics(run_dir, "cosine_similarity", rows,
                 ["image", "class", "variant", "cosine"])

    # ---- statistics ---------------------------------------------------------
    summary = {}
    for name in VARIANT_ORDER:
        values = np.array(per_variant[name])
        if values.size == 0:
            continue
        mean, lo, hi = bootstrap_ci(values, args.n_boot, seed=args.seed)
        summary[name] = {"mean": mean, "ci95": [lo, hi], "std": float(values.std(ddof=1)),
                         "n": int(values.size)}

    # Paired Wilcoxon against the transform that disturbs h least (excluding the
    # original, whose cosine is 1.0 by construction).
    comparisons = [n for n in summary if n != "original"]
    reference = max(comparisons, key=lambda n: summary[n]["mean"]) if comparisons else None
    if reference:
        try:
            from scipy.stats import wilcoxon

            raw_p = {}
            for name in comparisons:
                if name == reference:
                    continue
                stat, p = wilcoxon(per_variant[reference], per_variant[name])
                raw_p[name] = float(p)
            adjusted = holm_correction(raw_p)
            for name in raw_p:
                summary[name]["wilcoxon_vs_reference"] = {
                    "reference": reference,
                    "p_raw": raw_p[name],
                    "p_holm": adjusted[name],
                }
        except ImportError:
            print("  [stats] scipy not installed — skipping Wilcoxon tests")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure -------------------------------------------------------------
    import matplotlib.pyplot as plt

    names = [n for n in VARIANT_ORDER if n in summary and n != "original"]
    means = [summary[n]["mean"] for n in names]
    errors = np.array([[summary[n]["mean"] - summary[n]["ci95"][0] for n in names],
                       [summary[n]["ci95"][1] - summary[n]["mean"] for n in names]])

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(names, means, yerr=errors, capsize=3, color="#4C72B0")
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.set_ylabel(r"cosine$(h(T(x)),\ h(x))$")
    ax.set_ylim(min(means) - 0.1, 1.02)
    ax.set_title(f"E3 — {args.encoder} invariance, n={summary[names[0]]['n']} images/"
                 f"transform\nerror bars: bootstrap 95% CI")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.savefig(run_dir / "figures" / "invariance_cosine.png")

    # ---- (b) generator-visible invariance ----------------------------------
    if not args.skip_generation and args.n_visual > 0:
        visual_pairs = stratified_sample(args.test_dir, args.n_visual, seed=args.seed)
        for class_name, path in visual_pairs:
            img = Image.open(path).convert("RGB")
            variants = make_variants(img, args.crop_size, rng)

            # One noise draw, reused for every variant: differences between rows
            # are then attributable to h alone.
            noise = torch.randn(args.n_samples, 3, cfg["image_size"], cfg["image_size"],
                                device=device, generator=torch.Generator(device=device)
                                .manual_seed(args.seed))

            grid_rows = []
            for name, variant in variants.items():
                with torch.no_grad():
                    h = encoder(enc_transform(variant).unsqueeze(0).to(device))
                    generated = flow.sample(model, noise.clone(),
                                            h.repeat(args.n_samples, 1),
                                            num_steps=args.num_steps,
                                            cfg_scale=args.cfg_scale)
                thumb = TF.to_tensor(variant.resize((cfg["image_size"],) * 2))
                grid_rows.append(torch.cat([thumb.unsqueeze(0).to(device),
                                            unnorm(generated)], dim=0))

            out = run_dir / "figures" / f"variants_{class_name}_{path.stem}.png"
            vutils.save_image(torch.cat(grid_rows), out,
                              nrow=args.n_samples + 1, padding=2)
            print(f"  {out.name}  (rows: {', '.join(variants)})")

    print("\n  cosine by transform:")
    for name in names:
        s = summary[name]
        print(f"    {name:20s} {s['mean']:.4f}  [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]")
    print(f"\nDone — {run_dir}")


if __name__ == "__main__":
    main()
