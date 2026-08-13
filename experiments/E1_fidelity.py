"""
experiments/E1_fidelity.py — RQ1: is a frozen retinal SSL representation invertible?

Two complementary measurements, both over the same generated set:

  1. **Representation fidelity (the RCDM protocol).**  Generate one image per
     test representation, re-encode every generated image, and for each
     conditioning vector h_i rank all generated representations by distance to
     h_i.  If the generator is faithful, the image generated *from* h_i is the
     nearest one, i.e. rank 1.  Reported as mean rank and mean reciprocal rank
     (MRR), with a bootstrap CI over images.

     This is the metric that separates "the samples look like retinal images"
     from "the samples are the specific scan that h encodes".  Bordes et al.
     (2022) report MRR 0.97-0.99 for SSL encoders vs 0.69 for a supervised one
     on ImageNet; those values are the external reference point.

  2. **Distributional realism (FID).**  Frechet Inception Distance between the
     generated set and the corresponding real images.

     The real reference set is written through the *generator's own* training
     preprocessing (Resize + CenterCrop, see rcdm/dataset.py), not a plain
     resize.  Messidor-2 frames are roughly 3:2; squashing them to square would
     put a geometric distortion the generator never saw into the reference
     distribution and charge it to the model.

     Caveat to carry into the report: FID's Inception backbone is ImageNet-
     trained and poorly calibrated for retinal images, and with a few hundred
     images the estimator is biased.  Treat it as a relative signal between
     configurations, never as an absolute quality claim.

Uncertainty: MRR gets a bootstrap CI over the test set; run the script with
several --seed values and aggregate to get a spread on FID (a single FID number
carries no error bar and will not satisfy the report's figure requirements).

Usage:
    python experiments/E1_fidelity.py \
        --checkpoint models/jit_dinov3/final.pt \
        --encoder    dinov3 \
        --test_dir   data/raw/messidor2/test \
        --n_per_class 40 --cfg_scale 1.0 --num_steps 50 --seed 0 --device cuda

Output: metrics.csv (per-image ranks), summary.json (MRR/FID + CIs),
        figures/rank_distribution.png, and real/ + generated/ image folders.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
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

EXPERIMENT = "E1_fidelity"


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """(3, H, W) in [-1, 1] -> PIL RGB."""
    array = (unnorm(x).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def to_training_view(image: Image.Image, size: int) -> Image.Image:
    """
    Put a real image through the generator's own training preprocessing.

    This is byte-for-byte the pipeline in data/scripts/pack_dataset.py (bicubic
    Resize of the shorter side, then CenterCrop), which is what produced the
    packed tensor the generator was trained on.  It keeps the aspect ratio.

    Only the FID reference set needs this.  A plain resize to (size, size)
    anisotropically squashes Messidor-2's ~3:2 frames, so the reference
    distribution would carry a distortion the generator was never trained to
    reproduce and FID would charge the difference to the model.
    """
    resized = TF.resize(image, size, interpolation=TF.InterpolationMode.BICUBIC)
    return TF.center_crop(resized, size)


def rank_metrics(h_cond: torch.Tensor, h_gen: torch.Tensor) -> tuple:
    """
    Rank of the matching generation for every conditioning representation.

    For row i of h_cond, rank all rows of h_gen by cosine distance and report
    where h_gen[i] landed (1 = perfect fidelity).

    Cosine distance is used rather than L2 because representation norms vary
    considerably across images and the direction carries the semantics.

    Ties are resolved two ways so the reported number can never be an artefact
    of tie-breaking.  The optimistic rank counts only generations *strictly*
    closer than the matching one (a tie keeps rank 1); the pessimistic rank puts
    the matching generation last among its ties.  They are equal unless exact
    ties occur, which on continuous float32 representations should not happen —
    main() records the disagreement so "should not happen" is checked rather
    than assumed.

    Args:
        h_cond: (N, D) conditioning representations.
        h_gen:  (N, D) representations of the images generated from them.

    Returns:
        (ranks_optimistic, ranks_pessimistic), each a (N,) 1-based int array.
    """
    a = F.normalize(h_cond.float(), dim=1)
    b = F.normalize(h_gen.float(), dim=1)
    similarity = a @ b.T                                   # (N, N), higher = closer
    diagonal = similarity.diag().unsqueeze(1)              # (N, 1) self-similarity

    # Rank = 1 + how many other generations are closer than the matching one.
    optimistic = (similarity > diagonal).sum(dim=1).cpu().numpy() + 1
    # >= counts the matching generation itself, which supplies the +1, and every
    # tied competitor on top of it.
    pessimistic = (similarity >= diagonal).sum(dim=1).cpu().numpy()
    return optimistic, pessimistic


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI for a statistic over images."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = [statistic(values[rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(values)), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 — encoder-inversion fidelity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--test_dir", default="data/raw/messidor2/test")
    parser.add_argument("--n_per_class", type=int, default=40,
                        help="Images sampled per class. The rank metric is "
                             "computed against all of them jointly, so larger "
                             "is a harder and more meaningful test.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="1.0 = no guidance, and the default for every probe. "
                             "CFG extrapolates the conditioning signal, which "
                             "raises fidelity and suppresses diversity — the two "
                             "quantities E1 and E2 measure. Vary it only as an "
                             "explicit sensitivity analysis.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument("--skip_fid", action="store_true",
                        help="Skip FID (needs pytorch-fid and is the slow part)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)
    set_plot_style()

    model, flow, cfg = load_generator(args.checkpoint, device)
    encoder, enc_transform = load_probe_encoder(args.encoder, device, cfg, args.encoder_ckpt)

    pairs = stratified_sample(args.test_dir, args.n_per_class, seed=args.seed)
    print(f"  [data] {len(pairs)} images over {len(set(c for c, _ in pairs))} classes")

    run_dir = create_run_dir(
        EXPERIMENT, args.encoder, args.tag, args,
        extra={
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "model_cfg": cfg,
            "n_images": len(pairs),
        },
    )
    gen_dir = run_dir / "generated"
    real_dir = run_dir / "real"
    gen_dir.mkdir(exist_ok=True)
    real_dir.mkdir(exist_ok=True)

    # ---- generate one image per conditioning representation -----------------
    h_cond_all, h_gen_all, classes, stems = [], [], [], []

    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start: start + args.batch_size]
        images = [Image.open(p).convert("RGB") for _, p in batch]

        with torch.no_grad():
            x = torch.stack([enc_transform(im) for im in images]).to(device)
            h = encoder(x)
            noise = torch.randn(
                len(batch), 3, cfg["image_size"], cfg["image_size"], device=device
            )
            generated = flow.sample(
                model, noise, h, num_steps=args.num_steps, cfg_scale=args.cfg_scale
            )
            # Re-encode the generations through the *same* frozen encoder.
            gen_pils = [tensor_to_pil(g) for g in generated]
            x_gen = torch.stack([enc_transform(im) for im in gen_pils]).to(device)
            h_gen = encoder(x_gen)

        h_cond_all.append(h.cpu())
        h_gen_all.append(h_gen.cpu())
        for (class_name, path), gen_pil, real_img in zip(batch, gen_pils, images):
            stem = f"{class_name}_{path.stem}"
            gen_pil.save(gen_dir / f"{stem}.png")
            to_training_view(real_img, cfg["image_size"]).save(real_dir / f"{stem}.png")
            classes.append(class_name)
            stems.append(stem)

        print(f"  generated {min(start + args.batch_size, len(pairs))}/{len(pairs)}")

    h_cond = torch.cat(h_cond_all)
    h_gen = torch.cat(h_gen_all)

    # ---- rank fidelity ------------------------------------------------------
    ranks, ranks_pessimistic = rank_metrics(h_cond, h_gen)
    n_tied = int((ranks_pessimistic != ranks).sum())
    if n_tied:
        print(f"  [warn] {n_tied}/{len(ranks)} images have exact cosine ties — "
              f"read mrr_no_ties/top1_rate_no_ties, not mrr/top1_rate")
    reciprocal = 1.0 / ranks
    self_cos = F.cosine_similarity(h_cond.float(), h_gen.float(), dim=1).numpy()

    save_metrics(
        run_dir, "metrics",
        [
            {"image": s, "class": c, "rank": int(r), "rank_no_ties": int(rp),
             "reciprocal_rank": float(rr), "cosine_to_conditioning": float(cs)}
            for s, c, r, rp, rr, cs in zip(stems, classes, ranks, ranks_pessimistic,
                                           reciprocal, self_cos)
        ],
        ["image", "class", "rank", "rank_no_ties", "reciprocal_rank",
         "cosine_to_conditioning"],
    )

    mrr, mrr_lo, mrr_hi = bootstrap_ci(reciprocal, np.mean, args.n_boot, seed=args.seed)
    mean_rank, rank_lo, rank_hi = bootstrap_ci(ranks.astype(float), np.mean,
                                               args.n_boot, seed=args.seed)
    top1 = float((ranks == 1).mean())

    summary = {
        "n_images": len(ranks),
        "mrr": mrr, "mrr_ci95": [mrr_lo, mrr_hi],
        "mean_rank": mean_rank, "mean_rank_ci95": [rank_lo, rank_hi],
        "median_rank": float(np.median(ranks)),
        "top1_rate": top1,
        # Tie-pessimistic duplicates: identical to the above unless n_tied_images
        # > 0, in which case the headline numbers were flattered by tie-breaking.
        "n_tied_images": n_tied,
        "mrr_no_ties": float((1.0 / ranks_pessimistic).mean()),
        "top1_rate_no_ties": float((ranks_pessimistic == 1).mean()),
        # Chance levels for this N, so the numbers above can be read without
        # recomputing them: a random ordering gives mean rank (N+1)/2.
        "chance_mean_rank": (len(ranks) + 1) / 2,
        "chance_mrr": float(np.mean(1.0 / np.arange(1, len(ranks) + 1))),
        "chance_top1_rate": 1.0 / len(ranks),
        "mean_cosine_to_conditioning": float(self_cos.mean()),
        "cfg_scale": args.cfg_scale, "num_steps": args.num_steps, "seed": args.seed,
    }

    # ---- FID ----------------------------------------------------------------
    if not args.skip_fid:
        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths

            fid = calculate_fid_given_paths(
                [str(real_dir), str(gen_dir)], batch_size=min(50, len(pairs)),
                device=str(device), dims=2048,
            )
            summary["fid"] = float(fid)
        except ImportError:
            print("  [fid] pytorch-fid not installed — skipping")
            print(f"  [fid] run manually: python -m pytorch_fid {real_dir} {gen_dir}")
        except Exception as exc:  # noqa: BLE001 — FID must not lose the rank results
            print(f"  [fid] failed: {exc}")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure -------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(ranks, bins=min(50, len(set(ranks.tolist()))), color="#4C72B0")
    ax.set_xlabel(f"Rank of the matching generation (1 = perfect, {len(ranks)} = worst)")
    ax.set_ylabel("Images")
    ax.set_title(f"E1 fidelity — {args.encoder}\nMRR {mrr:.3f} "
                 f"[{mrr_lo:.3f}, {mrr_hi:.3f}], top-1 {top1:.1%}, n={len(ranks)}")
    fig.savefig(run_dir / "figures" / "rank_distribution.png")

    print("\n  " + "  ".join(f"{k}={v}" for k, v in summary.items() if not isinstance(v, list)))
    print(f"\nDone — {run_dir}")


if __name__ == "__main__":
    main()
