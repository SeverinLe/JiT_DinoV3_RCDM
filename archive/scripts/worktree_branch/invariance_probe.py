"""
scripts/invariance_probe.py

Probe the invariance of (a) the frozen DinoV3 backbone and (b) the JiT-RCDM
generator, using the augmented set produced by invariance_images.py.

For every grade folder in test_images_invariance/<grade>/ the script:

  1. Encodes the original + 5 augmented images with DinoV3 -> CLS token h (384).
  2. Measures backbone invariance: cosine similarity of each augmented h vs the
     original h. A perfectly invariant backbone would give 1.0 everywhere.
  3. Measures generator invariance: generates --n_samples images from each
     variant's h, reusing the SAME starting noise across all variants so any
     visual difference is attributable purely to h (not to the noise draw).
  4. Saves one comparison grid per grade: each row is a variant
     [labelled input | generated samples], the label shows the cosine
     similarity to the original CLS token.
  5. Writes cosine_similarity.csv and a printed summary.

Usage (from the worktree root, where data/ and checkpoints/ symlinks resolve):
    python scripts/invariance_probe.py \\
        --checkpoint checkpoints/jit_rcdm_step0100000.pt \\
        --inv_dir    test_images_invariance \\
        --out_dir    samples_invariance \\
        --n_samples  3 \\
        --cfg_scale  3.0 \\
        --num_steps  50 \\
        --device     cpu
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rcdm.encoder import load_encoder, build_transform, DINOV3_CHECKPOINT
from rcdm.jit import create_jit_model, FlowMatching

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def resolve_device(requested: str) -> torch.device:
    """Honour the requested device, but fall back to CPU if it is unavailable."""
    if requested == "mps" and not torch.backends.mps.is_available():
        print("  [device] mps requested but not available -> falling back to cpu")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("  [device] cuda requested but not available -> falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


# Fixed display order for the variants (rows of every grid).
VARIANT_ORDER = [
    "original",
    "crop128",
    "zoom_x2",
    "rot90",
    "mirror_vertical",
    "mirror_horizontal",
    "translate",
    "grayscale",
]


def load_model(checkpoint_path: str, device: torch.device):
    """Restore a JiT model from a checkpoint, applying EMA weights if present."""
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state["model_cfg"]
    model = create_jit_model(
        image_size=cfg["image_size"],
        patch_size=cfg.get("patch_size", 16),
        hidden_dim=cfg["hidden_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        h_dim=cfg["h_dim"],
        cond_dim=cfg.get("cond_dim"),
    )
    model.load_state_dict(state["model"])

    ema_state = state.get("ema")
    if ema_state is not None:
        shadow = ema_state.get("shadow", {})
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name].to(device))
        print("  [EMA] loaded EMA weights for inference")
    else:
        print("  [EMA] no EMA found in checkpoint - using raw weights")

    model.eval().to(device)
    return model, cfg, state


def unnorm(x: torch.Tensor) -> torch.Tensor:
    """Diffusion space [-1, 1] -> display space [0, 1]."""
    return (x.clamp(-1, 1) + 1.0) / 2.0


def label_thumbnail(img_pil: Image.Image, image_size: int, lines):
    """Resize a variant image to image_size and draw text labels on top of it."""
    thumb = img_pil.convert("RGB").resize((image_size, image_size))
    draw = ImageDraw.Draw(thumb)
    y = 2
    for text in lines:
        # cheap readability: black shadow then white text
        draw.text((3, y + 1), text, fill=(0, 0, 0))
        draw.text((2, y), text, fill=(255, 255, 0))
        y += 12
    t = transforms.ToTensor()(thumb)            # [0,1], (3, H, W)
    return t * 2.0 - 1.0                         # -> [-1, 1] to match generated


@torch.no_grad()
def encode(img_pil: Image.Image, encoder, enc_transform, device):
    x = enc_transform(img_pil).unsqueeze(0).to(device)
    out = encoder(pixel_values=x)
    return out.last_hidden_state[:, 0, :]        # (1, 384)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/jit_rcdm_step0100000.pt")
    parser.add_argument("--inv_dir", type=str, default="test_images_invariance",
                        help="Directory produced by invariance_images.py")
    parser.add_argument("--out_dir", type=str, default="samples_invariance")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Generated samples per variant")
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the shared starting noise")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--encoder_ckpt", type=str, default=DINOV3_CHECKPOINT)

    # Weights & Biases
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="jit-rcdm")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="invariance-probe")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Link to an existing W&B run (e.g. the training run)")
    args = parser.parse_args()

    use_wandb = (not args.no_wandb) and WANDB_AVAILABLE
    if not args.no_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Continuing without logging.")

    device = resolve_device(args.device)
    inv_dir = Path(args.inv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading JiT-RCDM from {args.checkpoint}...")
    model, cfg, state = load_model(args.checkpoint, device)
    flow = FlowMatching()
    image_size = cfg["image_size"]
    train_step = state.get("step", "unknown")
    print(f"  image_size={image_size}, cond_dim={cfg.get('cond_dim')}, "
          f"trained_steps={train_step}")

    print("Loading DinoV3 encoder...")
    encoder = load_encoder(device=device, checkpoint_path=args.encoder_ckpt)
    enc_transform = build_transform(image_size=224)  # DinoV3 fixed 14x14 grid

    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume="allow",
            config={
                "checkpoint": args.checkpoint,
                "inv_dir": str(inv_dir),
                "n_samples": args.n_samples,
                "num_steps": args.num_steps,
                "cfg_scale": args.cfg_scale,
                "seed": args.seed,
                "device": str(device),
                "trained_steps": train_step,
                **{f"model/{k}": v for k, v in cfg.items()},
            },
        )
        print(f"  W&B run: {run.url}")

    grades = sorted(p for p in inv_dir.iterdir() if p.is_dir())
    if not grades:
        raise SystemExit(f"No grade folders in {inv_dir}. Run invariance_images.py first.")

    # Shared starting noise: identical across every variant AND every grade so
    # differences are driven only by the conditioning vector h.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(args.n_samples, 3, image_size, image_size, generator=g).to(device)

    csv_rows = [["grade", "variant", "cosine_to_original", "l2_to_original"]]
    wandb_images = []
    wandb_table = wandb.Table(
        columns=["grade", "variant", "cosine_to_original", "l2_to_original"]
    ) if use_wandb else None

    for grade_dir in grades:
        grade = grade_dir.name
        print(f"\n=== {grade} ===")

        variants = [v for v in VARIANT_ORDER if (grade_dir / f"{v}.png").exists()]
        if "original" not in variants:
            print("  no original.png, skipping")
            continue

        # 1) encode every variant -> h
        pil_imgs, h_list = {}, {}
        for v in variants:
            pil = Image.open(grade_dir / f"{v}.png").convert("RGB")
            pil_imgs[v] = pil
            h_list[v] = encode(pil, encoder, enc_transform, device)  # (1, 384)

        h_orig = h_list["original"]

        # 2) backbone invariance: cosine + L2 of each h vs original
        rows_tensors = []
        for v in variants:
            cos = F.cosine_similarity(h_list[v], h_orig, dim=-1).item()
            l2 = (h_list[v] - h_orig).norm().item()
            csv_rows.append([grade, v, f"{cos:.4f}", f"{l2:.4f}"])
            print(f"  {v:<16s} cos={cos:+.4f}  l2={l2:7.3f}")
            if use_wandb:
                wandb_table.add_data(grade, v, cos, l2)

            # 3) generate n_samples from this variant's h, reusing shared noise
            h = h_list[v].expand(args.n_samples, -1)
            with torch.no_grad():
                gen = flow.sample(model, noise, h=h,
                                  num_steps=args.num_steps,
                                  cfg_scale=args.cfg_scale)        # (n, 3, H, W) [-1,1]

            # 4) build this row: [labelled input | generated samples]
            label = label_thumbnail(
                pil_imgs[v], image_size,
                [v, f"cos {cos:+.3f}"],
            ).unsqueeze(0).to(device)
            row = torch.cat([label, gen], dim=0)                    # (1+n, 3, H, W)
            rows_tensors.append(row)

        # one grid for the whole grade: rows stacked, each row = 1+n columns
        all_tiles = torch.cat(rows_tensors, dim=0)
        grid = vutils.make_grid(
            unnorm(all_tiles),
            nrow=args.n_samples + 1,
            padding=4,
            normalize=False,
        )
        out_path = out_dir / f"invariance_{grade}.png"
        vutils.save_image(grid, out_path)
        print(f"  saved grid -> {out_path}")

        if use_wandb:
            wandb_images.append(wandb.Image(
                grid.cpu(),
                caption=(f"{grade} | rows: {', '.join(variants)} | "
                         f"left: input (cos to original), right: {args.n_samples} "
                         f"generated (shared noise, cfg={args.cfg_scale})"),
            ))

    if use_wandb:
        wandb.log({"invariance_grids": wandb_images,
                   "cosine_similarity": wandb_table})
        wandb.finish()
        print(f"All grids + table logged to W&B: {run.url}")

    csv_path = out_dir / "cosine_similarity.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\nDone. Grids in {out_dir}/, similarities in {csv_path}")
    print("Row order per grid: " + ", ".join(VARIANT_ORDER))


if __name__ == "__main__":
    main()
