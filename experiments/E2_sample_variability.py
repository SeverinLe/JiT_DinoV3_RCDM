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
  lpips_between     mean pairwise perceptual distance, if lpips is installed

Reading the two SSIMs together separates the two failure modes: low
ssim_between means the representation underdetermines the image; high
ssim_between with low ssim_to_real means h pins down *an* image consistently,
just not the right one.

**The between-h baseline.**  None of the numbers above is interpretable on its
own.  "per-pixel std 0.08" is only meaningful against the std you would get with
no conditioning constraint at all, and on a domain this homogeneous — every
image a centre-cropped fundus photo with similar colour statistics — that floor
is high.  E1 made the same point in representation space: a generated image sits
at cosine 0.946 from its own h, but two *unrelated* retinas already sit at
0.921, so the raw number said almost nothing.

So every metric is also computed *between* conditioning vectors: one sample from
each of the N distinct h, compared the same way.  That gives

    ratio_within_over_between = within-h variability / between-h variability

which is the quantity to report.  Near 0 means h determines the image; near 1
means h constrains nothing the unconditional model would not have produced
anyway.  For SSIM the analogous normalisation is the fraction of the available
structural headroom that h explains,

    (ssim_within - ssim_between) / (1 - ssim_between)

The between-h per-pixel std is estimated from random subsets of the *same* size
k as the within-h estimate, so the two come from an identical estimator and the
ratio is not contaminated by sample-size bias.

Figures per conditioning image:
  samples_<stem>.png       [real | sample_1 ... sample_k]
  variability_<stem>.png   mean sample, per-pixel std map, |mean - real|

For a cross-class variance heatmap with a shared colour scale, see the companion
script E2_variance_heatmap.py.

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

# 8-bit quantisation step.  If the k samples differ by less than this they are
# byte-identical once saved, and no variability statistic computed from them
# means anything.
QUANTISATION_FLOOR = 1.0 / 255.0


def select_pairs(n: int, max_pairs: int | None = None, rng=None) -> list:
    """
    Unordered index pairs of a stack of n images, optionally subsampled.

    The between-h baseline compares ~30 images (435 pairs) and LPIPS is a
    network forward pass per pair, so capping the count keeps the baseline cheap
    without biasing the mean — pairs are exchangeable.
    """
    pairs = list(combinations(range(n), 2))
    if max_pairs and len(pairs) > max_pairs:
        rng = rng or np.random.default_rng(0)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]
    return pairs


def pairwise_ssim(images: np.ndarray, max_pairs: int | None = None, rng=None) -> float:
    """
    Mean SSIM over unordered pairs of a (k, H, W, 3) stack in [0, 1].

    Returns NaN if scikit-image is unavailable, so the rest of the metrics
    still get written.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")
    scores = [
        structural_similarity(images[i], images[j], channel_axis=2, data_range=1.0)
        for i, j in select_pairs(len(images), max_pairs, rng)
    ]
    return float(np.mean(scores)) if scores else float("nan")


_LPIPS_CACHE: dict = {}


def _lpips_model(device: torch.device):
    """Load AlexNet-LPIPS once per process; None if the package is absent."""
    if "net" not in _LPIPS_CACHE:
        try:
            import lpips as lpips_lib

            _LPIPS_CACHE["net"] = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()
            print("  [lpips] AlexNet backbone loaded")
        except ImportError:
            print("  [lpips] not installed — lpips columns will be NaN "
                  "(pip install lpips)")
            _LPIPS_CACHE["net"] = None
    return _LPIPS_CACHE["net"]


def pairwise_lpips(images: torch.Tensor, device: torch.device,
                   max_pairs: int | None = None, rng=None,
                   chunk: int = 16) -> float:
    """
    Mean pairwise LPIPS over a (k, 3, H, W) stack in [0, 1].

    SSIM is a local-statistics measure and saturates on images that differ in
    texture but not in layout; LPIPS compares deep features and is the better
    discriminator of "the vessels moved" versus "the noise changed".  It is
    optional because it pulls in an extra checkpoint download.

    Higher = more different, i.e. the opposite direction to SSIM.
    """
    net = _lpips_model(device)
    if net is None:
        return float("nan")

    pairs = select_pairs(len(images), max_pairs, rng)
    if not pairs:
        return float("nan")

    left = torch.stack([images[i] for i, _ in pairs])
    right = torch.stack([images[j] for _, j in pairs])
    scores = []
    with torch.no_grad():
        for start in range(0, len(pairs), chunk):
            # LPIPS expects [-1, 1], the same convention as the diffusion space.
            a = left[start: start + chunk].to(device) * 2.0 - 1.0
            b = right[start: start + chunk].to(device) * 2.0 - 1.0
            scores.append(net(a, b).flatten().cpu())
    return float(torch.cat(scores).mean())


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


def between_h_baseline(one_per_h: torch.Tensor, k: int, device: torch.device,
                       n_draws: int = 20, max_pairs: int = 200,
                       seed: int = 0) -> dict:
    """
    The same three metrics computed *across* conditioning vectors.

    Takes one sample from each of the N distinct h and measures how different
    those are from each other.  This is the no-conditioning-constraint floor:
    whatever variability survives here is what the generator produces anyway,
    independent of which h it was given.

    The std is averaged over ``n_draws`` random subsets of size k rather than
    computed once over all N, so it comes from the identical estimator as the
    within-h number and the ratio of the two is unbiased by sample size.

    Args:
        one_per_h: (N, 3, H, W) in [0, 1], one generation per conditioning image.
        k: the within-h sample count, matched here.
        n_draws: random size-k subsets averaged for the std estimate.
        max_pairs: cap on SSIM/LPIPS pair count.

    Returns:
        dict with per_pixel_std, ssim, lpips and the settings used.
    """
    rng = np.random.default_rng(seed)
    n = len(one_per_h)
    k = min(k, n)

    stds = []
    for _ in range(n_draws):
        subset = one_per_h[rng.choice(n, size=k, replace=False)]
        # Identical reduction to the within-h path: std over samples, then mean
        # over channels, then mean over pixels.
        stds.append(float(subset.std(dim=0).mean(dim=0).mean()))

    return {
        "per_pixel_std": float(np.mean(stds)),
        "per_pixel_std_sd_over_draws": float(np.std(stds, ddof=1)) if len(stds) > 1 else 0.0,
        "ssim": pairwise_ssim(to_numpy_stack(one_per_h), max_pairs, rng),
        "lpips": pairwise_lpips(one_per_h, device, max_pairs, rng),
        "n_images": n, "k": k, "n_draws": n_draws, "max_pairs": max_pairs,
    }


def nan_mean(values) -> float:
    """np.nanmean that returns NaN for an all-NaN column instead of warning.

    Reached whenever an optional metric is unavailable — lpips absent makes the
    whole column NaN, and that should not produce console noise.
    """
    finite = [v for v in np.asarray(values, dtype=float) if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def nan_std(values) -> float:
    """Sample std over the finite entries; 0.0 for a single value, NaN if none."""
    finite = [v for v in np.asarray(values, dtype=float) if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0


def safe_ratio(within: float, between: float) -> float:
    """within/between, NaN-safe — the reportable form of a variability metric."""
    if not np.isfinite(within) or not np.isfinite(between) or between == 0:
        return float("nan")
    return float(within / between)


def headroom_fraction(within: float, between: float) -> float:
    """
    Fraction of the available structural headroom that h explains.

    SSIM has a hard ceiling at 1, so the interesting quantity is not
    ssim_within - ssim_between but how much of the distance from the
    between-h floor to that ceiling the conditioning actually closes.
    1.0 = h fully determines the image, 0.0 = no better than an unrelated h.
    """
    if not np.isfinite(within) or not np.isfinite(between) or between >= 1.0:
        return float("nan")
    return float((within - between) / (1.0 - between))


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
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="1.0 = no guidance, and the default for every probe. "
                             "CFG > 1 suppresses sample diversity, which is "
                             "exactly what this experiment measures.")
    parser.add_argument("--n_baseline_draws", type=int, default=20,
                        help="Random size-k subsets averaged for the between-h "
                             "std estimate.")
    parser.add_argument("--max_baseline_pairs", type=int, default=200,
                        help="Cap on SSIM/LPIPS pairs in the between-h baseline.")
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

    # Bicubic to match data/scripts/pack_dataset.py, which produced the tensor
    # the generator was trained on.  ssim_to_real compares against this, so a
    # different resampling filter would show up as a fixed SSIM penalty.
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

    rows = []
    one_per_h = []          # first sample of each h — the between-h baseline stack
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
            "lpips_between": pairwise_lpips(samples.cpu(), device),
        }
        rows.append(metrics)
        one_per_h.append(samples[0].cpu())

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
                  "ssim_between", "ssim_to_real", "lpips_between"])

    # Aggregate per class — advanced grades have few training images, so a
    # per-class breakdown is where the data-scarcity effect shows up.
    metric_keys = ("per_pixel_std", "ssim_between", "ssim_to_real", "lpips_between")
    summary = {}
    for key in metric_keys:
        values = np.array([r[key] for r in rows], dtype=float)
        summary[key] = {
            "mean": nan_mean(values),
            "std": nan_std(values),
            "n": int(np.count_nonzero(np.isfinite(values))),
        }
    summary["per_class"] = {
        class_name: {
            key: nan_mean([r[key] for r in rows if r["class"] == class_name])
            for key in metric_keys
        }
        for class_name in sorted({r["class"] for r in rows})
    }

    # ---- between-h baseline: the floor every number above must be read against
    print(f"\n  [baseline] variability across {len(one_per_h)} different h ...")
    baseline = between_h_baseline(torch.stack(one_per_h), args.n_samples, device,
                                  n_draws=args.n_baseline_draws,
                                  max_pairs=args.max_baseline_pairs, seed=args.seed)
    summary["baseline_between_h"] = baseline

    within_std = summary["per_pixel_std"]["mean"]
    within_ssim = summary["ssim_between"]["mean"]
    within_lpips = summary["lpips_between"]["mean"]
    summary["normalised"] = {
        # 0 = h fully determines the image, 1 = h constrains nothing.
        "std_ratio_within_over_between": safe_ratio(within_std, baseline["per_pixel_std"]),
        "lpips_ratio_within_over_between": safe_ratio(within_lpips, baseline["lpips"]),
        # 1 = h fully determines the image, 0 = no better than an unrelated h.
        "ssim_headroom_explained": headroom_fraction(within_ssim, baseline["ssim"]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- collapse guard -----------------------------------------------------
    # A generator that ignores its noise maps each h to exactly one image.  Then
    # every ratio above goes to 0 and ssim_headroom_explained to 1, which reads
    # as "h determines the retina perfectly" — the precise opposite of what has
    # happened.  The premise of RQ2 (sample k images, see what varies) requires
    # a sampler that actually varies, so this is checked, not assumed.
    collapsed = bool(np.isfinite(within_std) and within_std < QUANTISATION_FLOOR)
    summary["sampler_collapsed"] = collapsed
    summary["quantisation_floor"] = QUANTISATION_FLOOR
    if collapsed:
        summary["normalised"] = {k: float("nan") for k in summary["normalised"]}
        print(f"\n  {'=' * 68}")
        print(f"  SAMPLER COLLAPSE: within-h std {within_std:.6f} is below the "
              f"8-bit\n  quantisation floor {QUANTISATION_FLOOR:.4f} — the k samples "
              "are byte-identical.")
        print("  This generator is a deterministic function of h, so it cannot "
              "report\n  what h discards.  The normalised ratios are voided "
              "rather than written,\n  because near-zero would otherwise read as "
              "'h determines everything'.")
        print(f"  {'=' * 68}")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    norm = summary["normalised"]
    print(f"\n  per-pixel std   within-h {within_std:.4f}  "
          f"between-h {baseline['per_pixel_std']:.4f}  "
          f"ratio {norm['std_ratio_within_over_between']:.3f}")
    print(f"  ssim            within-h {within_ssim:.4f}  "
          f"between-h {baseline['ssim']:.4f}  "
          f"headroom explained {norm['ssim_headroom_explained']:.3f}")
    print(f"  lpips           within-h {within_lpips:.4f}  "
          f"between-h {baseline['lpips']:.4f}  "
          f"ratio {norm['lpips_ratio_within_over_between']:.3f}")
    print("\n  ratio -> 0 means h determines the image; -> 1 means h constrains "
          "nothing.\n  Read these, not the raw numbers.")
    print(f"Done — {run_dir}")


if __name__ == "__main__":
    main()
