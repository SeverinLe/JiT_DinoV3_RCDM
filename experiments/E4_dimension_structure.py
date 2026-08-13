"""
experiments/E4_dimension_structure.py — RQ4: is the representation structured?

Bordes et al. (2022) found that individual dimensions of an SSL representation
carry separable, editable factors: take a query image, find its nearest
neighbours in representation space, identify the dimensions those neighbours
share, and that subset behaves like a handle on whatever the neighbourhood has
in common (in their ImageNet case, the background/scene signal).

This script reproduces that probe on retinal representations:

  1. **Select** the dimensions shared across the query's k-NN neighbourhood.
     Two heuristics, chosen with --selection:
        common_active  dimensions whose activation is consistently high across
                       the neighbourhood (the Bordes-style heuristic)
        low_variance   dimensions with the lowest variance across the
                       neighbourhood, i.e. what the neighbours agree on
  2. **Mask** them (set to zero) and generate — what disappears was carried by
     those dimensions.
  3. **Substitute** them with the values from a donor image and generate — if
     the factor transplants, the representation is compositional in that
     direction.

Interpretation is deliberately qualitative here; the quantitative counterpart is
E5, which feeds the same masked representations to a linear probe and asks
whether the visually-removed information was also the decodable information.
The selected dimension indices are written to dimensions.json for exactly that.

Caveat to carry into the report: this is a heuristic, not a discovery
procedure — the dimensions it finds depend on k, on the selection rule, and on
the encoder, and there is no guarantee that a "factor" aligns with coordinate
axes at all.

Usage:
    python experiments/E4_dimension_structure.py \
        --checkpoint models/jit_dinov3/final.pt \
        --encoder    dinov3 \
        --reps_file  data/processed/messidor2/dinov3/train_reps.pt \
        --n_queries  10 --k_neighbours 16 --n_dims 64 --seed 0 --device mps
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    unnorm,
)

EXPERIMENT = "E4_dimension_structure"


def select_dimensions(neighbourhood: torch.Tensor, n_dims: int,
                      selection: str) -> torch.Tensor:
    """
    Pick the dimensions that a k-NN neighbourhood has in common.

    Args:
        neighbourhood: (k, D) representations of the query and its neighbours.
        n_dims: how many dimensions to return.
        selection: "common_active" (high mean magnitude, low relative spread) or
            "low_variance" (smallest variance across the neighbourhood).

    Returns:
        (n_dims,) long tensor of dimension indices.
    """
    if selection == "low_variance":
        score = -neighbourhood.var(dim=0)                    # higher = more agreement
    elif selection == "common_active":
        mean = neighbourhood.mean(dim=0).abs()
        spread = neighbourhood.std(dim=0) + 1e-8
        score = mean / spread                                # consistently large
    else:
        raise ValueError(f"unknown selection rule {selection!r}")
    return torch.topk(score, k=n_dims).indices.sort().values


def generate_from(model, flow, h: torch.Tensor, noise: torch.Tensor,
                  num_steps: int, cfg_scale: float) -> torch.Tensor:
    """Generate with a fixed noise draw so rows differ only through h."""
    with torch.no_grad():
        return flow.sample(model, noise.clone(), h.repeat(noise.shape[0], 1),
                           num_steps=num_steps, cfg_scale=cfg_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="E4 — dimension-level structure")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--encoder_ckpt", default=None)
    parser.add_argument("--reps_file", default="data/processed/messidor2/dinov3/train_reps.pt",
                        help="Representation cache searched for nearest neighbours")
    parser.add_argument("--n_queries", type=int, default=10,
                        help="Query images. Bordes et al. showed one; ten keeps "
                             "the finding from being anecdotal.")
    parser.add_argument("--k_neighbours", type=int, default=16)
    parser.add_argument("--n_dims", type=int, default=64,
                        help="Dimensions to mask/substitute")
    parser.add_argument("--selection", default="common_active",
                        choices=["common_active", "low_variance"])
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Generations per condition")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="1.0 = no guidance, and the default for every probe. "
                             "CFG would amplify the effect of an edited h and "
                             "overstate how much each dimension controls.")
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

    cache = torch.load(args.reps_file, map_location="cpu", weights_only=False)
    reps, paths = cache["reps"].float(), cache["paths"]
    if cache.get("encoder") and cache["encoder"] != args.encoder:
        raise ValueError(f"{args.reps_file} was built with encoder "
                         f"'{cache['encoder']}', not '{args.encoder}'")
    if reps.shape[1] != cfg["h_dim"]:
        raise ValueError(f"cache has h_dim {reps.shape[1]}, generator expects {cfg['h_dim']}")
    print(f"  [cache] {reps.shape[0]} representations, D={reps.shape[1]}")

    run_dir = create_run_dir(
        EXPERIMENT, args.encoder, args.tag, args,
        extra={"checkpoint_sha256": sha256_file(args.checkpoint), "model_cfg": cfg,
               "n_reps": int(reps.shape[0])},
    )

    display_transform = transforms.Compose([
        transforms.Resize(image_size), transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    normalised = F.normalize(reps, dim=1)
    rng = np.random.default_rng(args.seed)
    query_indices = rng.choice(len(paths), size=min(args.n_queries, len(paths)),
                               replace=False)

    rows, selected_dims = [], {}
    for query_idx in query_indices:
        query_idx = int(query_idx)
        query_path = Path(paths[query_idx])

        # Nearest neighbours by cosine similarity, query included.
        similarity = normalised @ normalised[query_idx]
        neighbour_idx = torch.topk(similarity, k=args.k_neighbours + 1).indices
        neighbourhood = reps[neighbour_idx]

        dims = select_dimensions(neighbourhood, args.n_dims, args.selection)
        selected_dims[query_path.stem] = dims.tolist()

        # Donor: the *least* similar representation, so a substitution transplants
        # values from a genuinely different image rather than a near-duplicate.
        donor_idx = int(torch.argmin(similarity).item())
        donor_path = Path(paths[donor_idx])

        h_original = reps[query_idx: query_idx + 1].to(device)
        h_masked = h_original.clone()
        h_masked[:, dims] = 0.0
        h_substituted = h_original.clone()
        h_substituted[:, dims] = reps[donor_idx: donor_idx + 1, dims].to(device)

        noise = torch.randn(args.n_samples, 3, image_size, image_size, device=device)
        conditions = {
            "original": h_original,
            "masked": h_masked,
            "substituted": h_substituted,
        }

        grid_rows = []
        for name, h in conditions.items():
            generated = generate_from(model, flow, h, noise, args.num_steps, args.cfg_scale)
            grid_rows.append(unnorm(generated))
            # How far the edit moved the representation, for the record.
            rows.append({
                "query": query_path.stem,
                "condition": name,
                "donor": donor_path.stem if name == "substituted" else "",
                "cosine_to_original": float(
                    F.cosine_similarity(h.cpu(), h_original.cpu(), dim=1).item()
                ),
                "l2_to_original": float((h.cpu() - h_original.cpu()).norm().item()),
            })

        query_thumb = display_transform(
            Image.open(query_path).convert("RGB")
        ).unsqueeze(0).to(device)
        donor_thumb = display_transform(
            Image.open(donor_path).convert("RGB")
        ).unsqueeze(0).to(device)

        # Rows: [query | original samples], [query | masked], [donor | substituted]
        grid = torch.cat([
            torch.cat([query_thumb, grid_rows[0]], dim=0),
            torch.cat([query_thumb, grid_rows[1]], dim=0),
            torch.cat([donor_thumb, grid_rows[2]], dim=0),
        ], dim=0)
        out = run_dir / "figures" / f"dims_{query_path.stem}.png"
        vutils.save_image(grid, out, nrow=args.n_samples + 1, padding=2)
        print(f"  {out.name}  (rows: original, masked, substituted<-{donor_path.stem})")

    save_metrics(run_dir, "metrics", rows,
                 ["query", "condition", "donor", "cosine_to_original", "l2_to_original"])

    # E5 reads this file to test whether the visually-removed dimensions are also
    # the ones a linear probe depends on.
    (run_dir / "dimensions.json").write_text(json.dumps({
        "selection": args.selection,
        "n_dims": args.n_dims,
        "k_neighbours": args.k_neighbours,
        "encoder": args.encoder,
        "h_dim": int(reps.shape[1]),
        "per_query": selected_dims,
        "union": sorted({d for dims in selected_dims.values() for d in dims}),
    }, indent=2))

    print(f"\n  dimensions.json written ({args.n_dims} dims per query) — feed it to E5")
    print(f"Done — {run_dir}")


if __name__ == "__main__":
    main()
