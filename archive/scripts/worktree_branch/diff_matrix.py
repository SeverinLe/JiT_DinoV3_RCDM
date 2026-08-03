"""
scripts/diff_matrix.py

Pairwise difference maps for the N generations produced from a single
conditioning image, so the variant vs. consistent regions of the generator
become visible.

sampling.py / sampling_retfound.py / invariance_probe.py all produce
"[conditioning | gen_1 ... gen_N]" grids. Looking at those side by side tells
you the samples differ, but not *where*. This script builds an (N+1)x(N+1)
matrix over [original, gen_1, ..., gen_N] — the 5x5 case for the default
N=4 — where:

    diagonal          the image itself (original / generation), for reference
    upper triangle    |a - b| per-pixel absolute difference, mean over RGB
    lower triangle    1 - SSIM(a, b), i.e. *structural* dissimilarity
                      (--lower absdiff mirrors the abs-diff instead)

Both triangles use a shared colour scale across the whole matrix (auto vmax =
--vmax_pct percentile over all cells) so cells are directly comparable, and each
cell is annotated with its scalar metrics. Reading the matrix:

    row/col "original"    where the generations deviate from the real image
    gen_i vs gen_j block  where the generator is unstable across noise draws;
                          dark regions there = consistent structure that the
                          conditioning vector h pins down

A second figure (consistency_*.png) summarises the same information per pixel:
mean generation, per-pixel std across generations (the variability map), mean
|gen - original|, and a red overlay marking the most variant pixels.

Three input modes (pick exactly one):

  1. --cond_images  generate fresh samples from a checkpoint, same pipeline as
                    sampling.py (--encoder dinov3) / sampling_retfound.py
                    (--encoder retfound). Noise is seeded (--seed).
  2. --grid         slice an already-saved grid PNG from sampling.py or
                    invariance_probe.py (no checkpoint / encoder needed).
  3. --original + --generated   explicit image files.

Usage (from the worktree root, where data/ and checkpoints/ symlinks resolve):

    # 1) generate 4 samples and diff them
    python scripts/diff_matrix.py \\
        --checkpoint  checkpoints/jit_rcdm_step0100000.pt \\
        --cond_images data/messidor2/test/img1.png \\
        --out_dir     samples_diff \\
        --n_samples   4 --cfg_scale 3.0 --num_steps 50 --device cpu

    # 2) re-use an existing sampling.py grid (1 row, 1 + 4 tiles)
    python scripts/diff_matrix.py --grid samples/sample_img1.png --ncol 5

    # 3) a row of an invariance_probe.py grid (1 + 3 tiles, row 0 = original)
    python scripts/diff_matrix.py \\
        --grid samples_invariance/invariance_anodr.png --ncol 4 --row 0

Output (per input image):
    <out_dir>/diffmatrix_<stem>.png    the (N+1)x(N+1) matrix
    <out_dir>/consistency_<stem>.png   per-pixel variability summary
    <out_dir>/metrics_<stem>.csv       every pairwise metric
    plus W&B logging unless --no_wandb

Note: the first tile of an invariance_probe.py grid has the variant name and
cosine similarity drawn onto it, so in mode 3 that text shows up as a genuine
difference. Pass --original <clean file> to diff against the unlabelled image.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Colour maps
#
# matplotlib is not a dependency of this project, so the two perceptual maps we
# need are stored as anchor points and linearly interpolated. Close enough to
# matplotlib's inferno/viridis for reading a difference map.
# --------------------------------------------------------------------------- #
_CMAP_ANCHORS = {
    "inferno": [
        (0.001, 0.000, 0.014), (0.087, 0.045, 0.222), (0.229, 0.060, 0.439),
        (0.375, 0.121, 0.480), (0.517, 0.181, 0.468), (0.665, 0.246, 0.409),
        (0.808, 0.343, 0.302), (0.929, 0.507, 0.148), (0.988, 0.998, 0.645),
    ],
    "viridis": [
        (0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
        (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
        (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
        (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
    ],
    "gray": [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
}


def apply_cmap(x: torch.Tensor, name: str = "inferno") -> torch.Tensor:
    """(H, W) in [0, 1] -> (3, H, W) RGB in [0, 1]."""
    anchors = torch.tensor(_CMAP_ANCHORS[name], dtype=torch.float32, device=x.device)
    k = anchors.shape[0]
    pos = x.clamp(0.0, 1.0) * (k - 1)
    lo = pos.floor().long().clamp(max=k - 2)
    frac = (pos - lo).unsqueeze(-1)
    rgb = anchors[lo] * (1.0 - frac) + anchors[lo + 1] * frac   # (H, W, 3)
    return rgb.permute(2, 0, 1).contiguous()


# --------------------------------------------------------------------------- #
# SSIM (torch-only, no skimage dependency)
# --------------------------------------------------------------------------- #
def _gaussian_kernel(window: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window, dtype=torch.float32, device=device) - window // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim_map(a: torch.Tensor, b: torch.Tensor,
             data_range: float = 1.0, window: int = 11,
             sigma: float = 1.5) -> torch.Tensor:
    """
    Local SSIM between two (3, H, W) images in [0, 1].

    Returns the (H, W) SSIM map averaged over the colour channels; 1.0 means
    locally identical. Standard Gaussian-windowed formulation (Wang et al. 2004)
    with reflect padding so the map keeps the input resolution.
    """
    c = a.shape[0]
    pad = window // 2
    kernel = _gaussian_kernel(window, sigma, a.device).expand(c, 1, window, window)

    def blur(t: torch.Tensor) -> torch.Tensor:
        t = F.pad(t.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(t, kernel, groups=c).squeeze(0)

    mu_a, mu_b = blur(a), blur(b)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_aa = blur(a * a) - mu_aa
    sigma_bb = blur(b * b) - mu_bb
    sigma_ab = blur(a * b) - mu_ab

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_aa + mu_bb + c1) * (sigma_aa + sigma_bb + c2)
    return (num / den).mean(dim=0)


def smooth(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-blur a (H, W) map, so scattered per-pixel noise becomes regions."""
    if sigma <= 0:
        return m
    window = int(2 * round(3 * sigma) + 1)
    kernel = _gaussian_kernel(window, sigma, m.device)[None, None]
    pad = window // 2
    padded = F.pad(m[None, None], (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(padded, kernel)[0, 0]


def pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Scalar summary of one pair, plus the two per-pixel maps."""
    absdiff = (a - b).abs().mean(dim=0)              # (H, W), mean over RGB
    smap = ssim_map(a, b)                            # (H, W)
    mse = ((a - b) ** 2).mean().item()
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    return {
        "absdiff": absdiff,
        "dissim": (1.0 - smap).clamp(0.0, 1.0),
        "mae": absdiff.mean().item(),
        "rmse": math.sqrt(mse),
        "psnr": psnr,
        "ssim": smap.mean().item(),
    }


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def load_image(path: str, size: int) -> torch.Tensor:
    """
    File -> (3, size, size) in [0, 1].

    Resize + CenterCrop matches the diffusion_transform of sampling.py, so the
    original is framed exactly like the images the model was conditioned on.
    """
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    return tf(Image.open(path).convert("RGB"))


def split_grid(path: str, ncol: int, padding: int, row: int):
    """
    Slice a make_grid() PNG back into its tiles.

    make_grid lays tiles out as: x = padding + col * (tile + padding), same for
    y, so with the column count known the tile size is exact. Returns the tiles
    of the requested row as a list of (3, tile, tile) tensors in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    span, rem = divmod(w - padding, ncol)
    if rem != 0 or span - padding <= 0:
        raise SystemExit(
            f"{path}: width {w} is not consistent with --ncol {ncol} "
            f"--padding {padding}. Pass the column count of the saved grid "
            f"(n_samples + 1)."
        )
    tile = span - padding
    nrow = (h - padding) // span
    if not 0 <= row < nrow:
        raise SystemExit(f"{path}: --row {row} out of range (grid has {nrow} rows)")

    to_tensor = transforms.ToTensor()
    tiles = []
    for col in range(ncol):
        x = padding + col * span
        y = padding + row * span
        tiles.append(to_tensor(img.crop((x, y, x + tile, y + tile))))
    return tiles


def load_generator(args, device):
    """Lazy model/encoder construction — only needed for --cond_images."""
    from rcdm.jit import create_jit_model, FlowMatching

    if args.encoder == "retfound":
        from rcdm.encoder_retfound import (load_encoder, build_transform,
                                           RETFOUND_CHECKPOINT)
        default_ckpt = RETFOUND_CHECKPOINT
    else:
        from rcdm.encoder import load_encoder, build_transform, DINOV3_CHECKPOINT
        default_ckpt = DINOV3_CHECKPOINT

    state = torch.load(args.checkpoint, map_location=device)
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

    encoder = load_encoder(device=device,
                           checkpoint_path=args.encoder_ckpt or default_ckpt)
    # Both backbones have fixed pos-embeds for a 14x14 patch grid -> 224 px.
    enc_transform = build_transform(image_size=224)
    return model, FlowMatching(), cfg, state, encoder, enc_transform


@torch.no_grad()
def generate(args, img_path, model, flow, cfg, encoder, enc_transform, device):
    """Encode one conditioning image and sample n_samples from its h."""
    image_size = cfg["image_size"]
    pil = Image.open(img_path).convert("RGB")
    x_enc = enc_transform(pil).unsqueeze(0).to(device)

    if args.encoder == "retfound":
        h_single = encoder.forward_features(x_enc)[:, 0, :]      # timm API
    else:
        h_single = encoder(pixel_values=x_enc).last_hidden_state[:, 0, :]
    h = h_single.expand(args.n_samples, -1)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(args.n_samples, 3, image_size, image_size,
                        generator=g).to(device)
    gen = flow.sample(model, noise, h=h,
                      num_steps=args.num_steps, cfg_scale=args.cfg_scale)
    gen = (gen.clamp(-1, 1) + 1.0) / 2.0                         # -> [0, 1]
    return gen.cpu().unbind(0)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
)


def get_font(size: int):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def text(draw, xy, s, font, fill=(255, 255, 255), shadow=True):
    """Text with a 1 px black shadow so it stays readable over any heat map."""
    x, y = xy
    if shadow:
        draw.text((x + 1, y + 1), s, font=font, fill=(0, 0, 0))
    draw.text((x, y), s, font=font, fill=fill)


def to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) in [0, 1] -> PIL."""
    arr = (t.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def resize_map(m: torch.Tensor, size: int) -> torch.Tensor:
    """Resample a (H, W) map to (size, size), antialiased when downscaling."""
    if m.shape[-1] == size and m.shape[-2] == size:
        return m
    return F.interpolate(m[None, None], size=(size, size),
                         mode="bilinear", align_corners=False,
                         antialias=True)[0, 0]


def heat_tile(m: torch.Tensor, vmax: float, cmap: str, size: int) -> Image.Image:
    return to_pil(apply_cmap(resize_map(m, size) / max(vmax, 1e-8), cmap))


def image_tile(t: torch.Tensor, size: int) -> Image.Image:
    return to_pil(F.interpolate(t[None], size=(size, size), mode="bilinear",
                                align_corners=False, antialias=True)[0])


def fmt_tick(v: float) -> str:
    """Difference magnitudes span orders of magnitude; %.3f would print 0.000."""
    return f"{v:.3f}" if v >= 0.01 else f"{v:.1e}"


def draw_colorbar(canvas, draw, x, y, w, h, cmap, vmax, font, label):
    ramp = torch.linspace(0.0, 1.0, w).unsqueeze(0).expand(h, w)
    canvas.paste(to_pil(apply_cmap(ramp, cmap)), (x, y))
    draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(90, 90, 90))
    text(draw, (x, y - 18), label, font, shadow=False)
    text(draw, (x, y + h + 4), "0", font, shadow=False)
    mid, hi = fmt_tick(vmax / 2), fmt_tick(vmax)
    text(draw, (x + w // 2 - draw.textlength(mid, font=font) / 2, y + h + 4),
         mid, font, shadow=False)
    text(draw, (x + w - draw.textlength(hi, font=font), y + h + 4),
         hi, font, shadow=False)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
BG = (24, 24, 28)
FG = (235, 235, 235)


def cell_group(i: int, j: int) -> str:
    """
    Which comparison family a cell belongs to.

    Generations typically sit far closer to each other than to the real image,
    so the two families get their own colour scale by default (--norm group).
    Sharing one scale would flatten the whole gen<->gen block to black.
    """
    return "orig" if (i == 0 or j == 0) else "gen"


def build_matrix(images, labels, results, scales, args, subtitle):
    """
    The (N+1)x(N+1) figure.

    results[(i, j)] holds the metrics/maps for every i != j; cell (i, i) shows
    image i so each row and column is anchored to a visible reference.
    scales maps ("abs"|"dis", "orig"|"gen") -> vmax for the colour mapping.
    """
    n = len(images)
    cell, gap = args.cell_size, 8
    pad_l, pad_t, pad_r, pad_b = 118, 96, 24, 142
    font_lbl, font_cell, font_title = get_font(17), get_font(14), get_font(24)

    note1 = "diagonal = the image itself | bright = differs, dark = consistent"
    note2 = ("each family has its own colour scale (--norm group): the "
             "gen <-> gen block is stretched to its own range, so do not compare "
             "its brightness against the original row/column"
             if args.norm == "group" else
             "one shared colour scale across every cell (--norm global)")

    # the footnotes are wider than a small matrix — grow the canvas to fit them
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    legend_w = max(probe.textlength(s, font=font_cell) for s in (note1, note2))
    width = max(pad_l + n * cell + (n - 1) * gap + pad_r, int(legend_w) + 48)
    height = pad_t + n * cell + (n - 1) * gap + pad_b
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    text(draw, (24, 18), "Pairwise difference matrix", font_title, FG, shadow=False)
    text(draw, (24, 50), subtitle, font_lbl, (170, 170, 175), shadow=False)

    def cx(j):
        return pad_l + j * (cell + gap)

    def cy(i):
        return pad_t + i * (cell + gap)

    for k, name in enumerate(labels):
        text(draw, (cx(k) + 4, pad_t - 22), name, font_lbl, FG, shadow=False)
        text(draw, (12, cy(k) + cell // 2 - 8), name, font_lbl, FG, shadow=False)

    for i in range(n):
        for j in range(n):
            x, y = cx(j), cy(i)
            if i == j:
                canvas.paste(image_tile(images[i], cell), (x, y))
                draw.rectangle([x, y, x + cell - 1, y + cell - 1],
                               outline=(250, 200, 60), width=2)
                text(draw, (x + 6, y + cell - 24), labels[i], font_cell)
                continue

            r = results[(i, j)]
            grp = cell_group(i, j)
            if j > i or args.lower == "absdiff":
                tile = heat_tile(r["absdiff"], scales[("abs", grp)], args.cmap, cell)
                lines = [f"MAE {r['mae']:.4f}", f"PSNR {r['psnr']:.1f} dB"]
            else:
                tile = heat_tile(r["dissim"], scales[("dis", grp)], args.cmap, cell)
                lines = [f"1-SSIM {1 - r['ssim']:.4f}", f"SSIM {r['ssim']:.4f}"]
            canvas.paste(tile, (x, y))
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(70, 70, 76))
            for row, s in enumerate(lines):
                text(draw, (x + 6, y + 6 + row * 17), s, font_cell)

    # one colour bar per (metric, comparison family) that is actually on screen
    metrics = [("abs", "|a - b|")]
    if args.lower == "ssim":
        metrics.append(("dis", "1 - SSIM"))
    families = ([("orig", "vs original"), ("gen", "gen <-> gen")]
                if args.norm == "group" else [("orig", "all cells")])
    bars = [(scales[(m, f)], f"{m_name}  {f_name}")
            for m, m_name in metrics for f, f_name in families]

    bar_y = height - pad_b + 42
    bar_gap = 40
    bar_w = min(280, (width - 2 * pad_l - (len(bars) - 1) * bar_gap) // len(bars))
    for k, (vmax, label) in enumerate(bars):
        draw_colorbar(canvas, draw, pad_l + k * (bar_w + bar_gap), bar_y,
                      bar_w, 16, args.cmap, vmax, font_cell, label)

    text(draw, (24, height - 48), note1, font_cell, (170, 170, 175), shadow=False)
    text(draw, (24, height - 28), note2, font_cell, (170, 170, 175), shadow=False)
    return canvas


def build_consistency(original, gens, args, subtitle):
    """
    Per-pixel summary over the N generations:
    original | mean | std (variability) | mean |gen - original| | variant overlay.
    """
    stack = torch.stack(gens)                       # (N, 3, H, W)
    mean = stack.mean(dim=0)
    std = stack.std(dim=0, unbiased=False).mean(dim=0)                 # (H, W)
    dev = (stack - original.unsqueeze(0)).abs().mean(dim=(0, 1))       # (H, W)

    vmax_std = max(torch.quantile(std.flatten(), args.vmax_pct / 100.0).item(), 1e-8)
    vmax_dev = max(torch.quantile(dev.flatten(), args.vmax_pct / 100.0).item(), 1e-8)

    # Grayscale original with the most-variant *regions* tinted red. The std map
    # is blurred first: thresholding it raw just marks scattered single pixels,
    # which says nothing about which anatomy the generator is unsure of.
    std_regions = smooth(std, args.overlay_sigma)
    thresh = torch.quantile(std_regions.flatten(), args.overlay_pct / 100.0).item()
    mask = (std_regions >= thresh).float()
    gray = original.mean(dim=0, keepdim=True).expand(3, -1, -1)
    alpha = 0.6 * mask
    overlay = gray * (1 - alpha) + torch.tensor([1.0, 0.15, 0.15]).view(3, 1, 1) * alpha

    cell, gap = args.cell_size, 8
    pad_l, pad_t, pad_b = 24, 96, 108
    panels = [
        (image_tile(original, cell), "original"),
        (image_tile(mean, cell), f"mean of {len(gens)} generations"),
        (heat_tile(std, vmax_std, args.cmap, cell), "per-pixel std (variability)"),
        (heat_tile(dev, vmax_dev, args.cmap, cell), "mean |gen - original|"),
        (image_tile(overlay, cell),
         f"top {100 - args.overlay_pct:.0f}% most variant regions"),
    ]

    font_lbl, font_cell, font_title = get_font(17), get_font(14), get_font(24)
    note = ("dark in the std map = every generation agrees there (structure fixed "
            "by h); bright = free to vary with the noise draw")
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = max(pad_l * 2 + len(panels) * cell + (len(panels) - 1) * gap,
                int(probe.textlength(note, font=font_cell)) + 48)
    height = pad_t + cell + pad_b
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    text(draw, (24, 18), "Generation consistency", font_title, FG, shadow=False)
    text(draw, (24, 50), subtitle, font_lbl, (170, 170, 175), shadow=False)

    for k, (tile, name) in enumerate(panels):
        x = pad_l + k * (cell + gap)
        canvas.paste(tile, (x, pad_t))
        draw.rectangle([x, pad_t, x + cell - 1, pad_t + cell - 1],
                       outline=(70, 70, 76))
        text(draw, (x + 2, pad_t - 22), name, font_lbl, FG, shadow=False)

    bar_y = pad_t + cell + 40
    bar_w = min(cell, 300)
    draw_colorbar(canvas, draw, pad_l + 2 * (cell + gap), bar_y, bar_w, 16,
                  args.cmap, vmax_std, font_cell, "std")
    draw_colorbar(canvas, draw, pad_l + 3 * (cell + gap), bar_y, bar_w, 16,
                  args.cmap, vmax_dev, font_cell, "mean abs. deviation")
    text(draw, (24, height - 26), note, font_cell, (170, 170, 175), shadow=False)

    return canvas, {"std_mean": std.mean().item(),
                    "std_vmax": vmax_std,
                    "dev_mean": dev.mean().item()}


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Pairwise difference-map matrix over the generations of one "
                    "conditioning image.")

    src = parser.add_argument_group("input (choose exactly one mode)")
    src.add_argument("--cond_images", type=str, nargs="+",
                     help="Generate samples from these conditioning images "
                          "(requires --checkpoint)")
    src.add_argument("--grid", type=str, nargs="+",
                     help="Slice tiles out of saved sampling.py / "
                          "invariance_probe.py grid PNGs")
    src.add_argument("--generated", type=str, nargs="+",
                     help="Explicit generated image files (requires --original)")
    src.add_argument("--original", type=str, default=None,
                     help="Original / conditioning image. Required with "
                          "--generated, optional override for --grid.")

    gen = parser.add_argument_group("generation mode")
    gen.add_argument("--checkpoint", type=str, default=None)
    gen.add_argument("--encoder", type=str, default="dinov3",
                     choices=["dinov3", "retfound"])
    gen.add_argument("--encoder_ckpt", type=str, default=None)
    gen.add_argument("--n_samples", type=int, default=4)
    gen.add_argument("--num_steps", type=int, default=50)
    gen.add_argument("--cfg_scale", type=float, default=3.0)
    gen.add_argument("--seed", type=int, default=0,
                     help="Seed for the starting noise (reproducible matrices)")

    grd = parser.add_argument_group("grid mode")
    grd.add_argument("--ncol", type=int, default=5,
                     help="Columns in the saved grid (n_samples + 1)")
    grd.add_argument("--padding", type=int, default=4,
                     help="make_grid padding used when the grid was saved")
    grd.add_argument("--row", type=int, default=0,
                     help="Row to analyse in a multi-row grid")

    out = parser.add_argument_group("output")
    out.add_argument("--out_dir", type=str, default="samples_diff")
    out.add_argument("--cell_size", type=int, default=224)
    out.add_argument("--cmap", type=str, default="inferno",
                     choices=sorted(_CMAP_ANCHORS))
    out.add_argument("--lower", type=str, default="ssim",
                     choices=["ssim", "absdiff"],
                     help="Lower triangle metric (upper is always |a - b|)")
    out.add_argument("--norm", type=str, default="group",
                     choices=["group", "global"],
                     help="group: scale 'vs original' and 'gen <-> gen' cells "
                          "separately (default, keeps the gen block readable "
                          "when the generator is stable). "
                          "global: one scale for the whole matrix.")
    out.add_argument("--vmax", type=float, default=None,
                     help="Fix the abs-diff colour scale instead of auto")
    out.add_argument("--vmax_pct", type=float, default=99.0,
                     help="Percentile used for the automatic colour scale")
    out.add_argument("--overlay_pct", type=float, default=80.0,
                     help="Pixels above this std percentile are marked variant")
    out.add_argument("--overlay_sigma", type=float, default=2.0,
                     help="Gaussian sigma used to turn the std map into regions "
                          "before thresholding (0 = threshold raw pixels)")
    out.add_argument("--device", type=str, default="cpu")

    wb = parser.add_argument_group("weights & biases")
    wb.add_argument("--no_wandb", action="store_true")
    wb.add_argument("--wandb_project", type=str, default="jit-rcdm")
    wb.add_argument("--wandb_entity", type=str, default=None)
    wb.add_argument("--wandb_run_name", type=str, default="diff-matrix")
    wb.add_argument("--wandb_run_id", type=str, default=None)

    args = parser.parse_args()

    modes = [bool(args.cond_images), bool(args.grid), bool(args.generated)]
    if sum(modes) != 1:
        parser.error("pass exactly one of --cond_images, --grid, --generated")
    if args.cond_images and not args.checkpoint:
        parser.error("--cond_images requires --checkpoint")
    if args.generated and not args.original:
        parser.error("--generated requires --original")

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = (not args.no_wandb) and WANDB_AVAILABLE
    if not args.no_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Continuing without logging.")

    # ---------------------------------------------------------------- #
    # Collect (stem, original, [generations]) for every requested item
    # ---------------------------------------------------------------- #
    items = []
    subtitle_common = ""

    if args.cond_images:
        print(f"Loading JiT-RCDM from {args.checkpoint} ...")
        model, flow, cfg, state, encoder, enc_tf = load_generator(args, device)
        train_step = state.get("step", "unknown")
        image_size = cfg["image_size"]
        print(f"  image_size={image_size}, h_dim={cfg['h_dim']}, "
              f"cond_dim={cfg.get('cond_dim')}, trained_steps={train_step}")
        subtitle_common = (f"{args.encoder} | {args.num_steps}-step Heun | "
                           f"cfg={args.cfg_scale} | seed={args.seed} | "
                           f"trained_steps={train_step}")
        for path in args.cond_images:
            print(f"\nConditioning on {path} -> {args.n_samples} samples ...")
            gens = generate(args, path, model, flow, cfg, encoder, enc_tf, device)
            items.append((Path(path).stem, load_image(path, image_size), list(gens)))

    elif args.grid:
        subtitle_common = f"tiles sliced from saved grid (row {args.row})"
        for path in args.grid:
            tiles = split_grid(path, args.ncol, args.padding, args.row)
            original = (load_image(args.original, tiles[0].shape[-1])
                        if args.original else tiles[0])
            items.append((Path(path).stem, original, tiles[1:]))

    else:
        # everything is framed to the shorter side of the first generation
        first = Image.open(args.generated[0])
        size = min(first.size)
        gens = [load_image(p, size) for p in args.generated]
        items.append((Path(args.original).stem, load_image(args.original, size), gens))
        subtitle_common = "explicit image files"

    # ---------------------------------------------------------------- #
    # Per item: metrics -> figures -> csv
    # ---------------------------------------------------------------- #
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name, id=args.wandb_run_id, resume="allow",
            config={k: v for k, v in vars(args).items() if v is not None},
        )
        print(f"  W&B run: {wandb_run.url}")
    wandb_images, wandb_table = [], (
        wandb.Table(columns=["item", "a", "b", "mae", "rmse", "psnr", "ssim"])
        if use_wandb else None)

    for stem, original, gens in items:
        n_gen = len(gens)
        if n_gen < 2:
            print(f"  {stem}: need at least 2 generations, got {n_gen} - skipping")
            continue

        size = original.shape[-1]
        gens = [g if g.shape[-1] == size else
                F.interpolate(g[None], size=(size, size), mode="bilinear",
                              align_corners=False, antialias=True)[0]
                for g in gens]
        images = [original] + gens
        labels = ["original"] + [f"gen {i + 1}" for i in range(n_gen)]

        results, csv_rows = {}, [["a", "b", "mae", "rmse", "psnr_db", "ssim"]]
        for i in range(len(images)):
            for j in range(len(images)):
                if i == j:
                    continue
                if (j, i) in results:            # metrics are symmetric
                    results[(i, j)] = results[(j, i)]
                    continue
                r = pair_metrics(images[i], images[j])
                results[(i, j)] = r
                csv_rows.append([labels[i], labels[j], f"{r['mae']:.6f}",
                                 f"{r['rmse']:.6f}", f"{r['psnr']:.3f}",
                                 f"{r['ssim']:.6f}"])
                if use_wandb:
                    wandb_table.add_data(stem, labels[i], labels[j], r["mae"],
                                         r["rmse"], r["psnr"], r["ssim"])

        # Colour scales. Cells are only comparable within one scale, so the
        # "vs original" and "gen <-> gen" families are scaled separately unless
        # --norm global was asked for (see cell_group).
        q = args.vmax_pct / 100.0
        scales = {}
        for key, field in (("abs", "absdiff"), ("dis", "dissim")):
            for family in ("orig", "gen"):
                sel = [r[field] for (i, j), r in results.items()
                       if args.norm == "global" or cell_group(i, j) == family]
                v = torch.quantile(torch.stack(sel).flatten(), q).item()
                if key == "abs" and args.vmax is not None:
                    v = args.vmax
                scales[(key, family)] = max(v, 1e-6)

        subtitle = f"{stem} | {n_gen} generations | {subtitle_common}"
        matrix = build_matrix(images, labels, results, scales, args, subtitle)
        consistency, stats = build_consistency(original, gens, args, subtitle)

        m_path = out_dir / f"diffmatrix_{stem}.png"
        c_path = out_dir / f"consistency_{stem}.png"
        csv_path = out_dir / f"metrics_{stem}.csv"
        matrix.save(m_path)
        consistency.save(c_path)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows(csv_rows)

        # console summary: how much of the variation is model instability vs
        # deviation from the real image
        vs_orig = [results[(0, k)]["mae"] for k in range(1, len(images))]
        pairs = [results[(a, b)]["mae"]
                 for a in range(1, len(images)) for b in range(a + 1, len(images))]
        ssim_orig = [results[(0, k)]["ssim"] for k in range(1, len(images))]
        ssim_pair = [results[(a, b)]["ssim"]
                     for a in range(1, len(images)) for b in range(a + 1, len(images))]
        print(f"\n=== {stem} ===")
        print(f"  gen vs original : MAE {sum(vs_orig) / len(vs_orig):.4f}   "
              f"SSIM {sum(ssim_orig) / len(ssim_orig):.4f}")
        print(f"  gen vs gen      : MAE {sum(pairs) / len(pairs):.4f}   "
              f"SSIM {sum(ssim_pair) / len(ssim_pair):.4f}")
        print(f"  per-pixel std   : mean {stats['std_mean']:.4f}  "
              f"p{args.vmax_pct:g} {stats['std_vmax']:.4f}")
        mean_pair = sum(pairs) / len(pairs)
        if mean_pair < 1.0 / 255:
            print(f"  [!] gen <-> gen MAE {mean_pair:.5f} is below the 8-bit "
                  f"quantisation step (1/255 = 0.0039): these generations are "
                  f"effectively identical, and the gen <-> gen cells show "
                  f"rounding dust, not model variability.")
        print(f"  saved -> {m_path}")
        print(f"           {c_path}")
        print(f"           {csv_path}")

        if use_wandb:
            wandb_images += [
                wandb.Image(str(m_path), caption=f"{stem} | difference matrix"),
                wandb.Image(str(c_path), caption=f"{stem} | consistency"),
            ]

    if use_wandb:
        wandb.log({"diff_matrix": wandb_images, "pairwise_metrics": wandb_table})
        wandb.finish()
        print(f"\nLogged to W&B: {wandb_run.url}")

    print("\nDone.")


if __name__ == "__main__":
    main()
