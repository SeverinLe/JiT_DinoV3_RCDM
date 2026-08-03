"""
experiments/common.py

Shared plumbing for the E1-E5 probes.

Every experiment script uses the same three things, and getting them consistent
is what lets the report claim reproducibility:

  1. ``create_run_dir()`` — one timestamped directory per run under
     experiments/results/<experiment>/<encoder>/<tag>/, containing figures/,
     metrics files, and a run_config.json written *before* any compute starts.
  2. ``run_config.json`` — every CLI argument, the RNG seed, the git commit, the
     sha256 of the model checkpoint, and library versions.  A figure in the
     report can therefore always be traced to the exact state that produced it.
  3. ``load_generator()`` / ``load_probe_encoder()`` — one loading path, with EMA
     weights applied and the encoder cross-checked against the checkpoint's
     recorded h_dim, so an experiment cannot silently pair a model with the wrong
     representation space.

Import from an experiment script as:

    from common import (create_run_dir, load_generator, load_probe_encoder,
                        resolve_device, set_seed, save_metrics)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

sys.path.insert(0, str(REPO_ROOT))

from rcdm.encoders import build_transform, get_encoder, get_h_dim  # noqa: E402
from rcdm.jit import FlowMatching, create_jit_model  # noqa: E402


# ---------------------------------------------------------------------------
# Determinism and devices
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed torch (CPU + all accelerators) and Python's hash-independent RNGs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Honour the requested device, falling back to CPU with a printed warning."""
    if requested == "mps" and not torch.backends.mps.is_available():
        print("  [device] mps unavailable -> cpu")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("  [device] cuda unavailable -> cpu")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 — checkpoints are too large to read into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Current commit hash, with a -dirty suffix if the tree has changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"{commit}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _versions() -> dict:
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform()}
    for name in ("torchvision", "timm", "transformers", "numpy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 — a missing optional dep is not an error here
            versions[name] = None
    return versions


def create_run_dir(experiment: str, encoder: str, tag: str | None = None,
                   args=None, extra: dict | None = None) -> Path:
    """
    Create experiments/results/<experiment>/<encoder>/<tag>/ and write its manifest.

    Args:
        experiment: e.g. "E3_invariance" — must match the script name.
        encoder: registry name of the encoder under study.
        tag: run label; defaults to a UTC timestamp.
        args: the argparse Namespace, recorded verbatim.
        extra: anything else worth pinning (checkpoint hash, dataset sizes, ...).

    Returns:
        Path to the run directory, with figures/ already created.
    """
    tag = tag or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / experiment / encoder / tag
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": experiment,
        "encoder": encoder,
        "tag": tag,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "args": vars(args) if args is not None else {},
        "versions": _versions(),
    }
    if extra:
        manifest.update(extra)
    (run_dir / "run_config.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"  [run] {run_dir.relative_to(REPO_ROOT)}")
    return run_dir


def save_metrics(run_dir: Path, name: str, rows: list, fieldnames: list) -> Path:
    """Write a list of dicts to <run_dir>/<name>.csv and return the path."""
    import csv

    out = Path(run_dir) / f"{name}.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [metrics] {out.name} ({len(rows)} rows)")
    return out


# ---------------------------------------------------------------------------
# Model / encoder loading
# ---------------------------------------------------------------------------

def load_generator(checkpoint_path, device: torch.device, use_ema: bool = True):
    """
    Restore a trained JiT-RCDM generator.

    Architecture comes from the checkpoint's own ``model_cfg``, never from CLI
    flags, so a checkpoint can always be loaded without knowing how it was made.

    EMA weights are applied by default: the JiT paper (Tab. 9) reports EMA at
    decay 0.9999 as the best-FID configuration, and every reported number should
    come from the same weights.

    Returns:
        (model, flow, cfg) — model in eval mode on ``device``.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_cfg" not in state:
        raise ValueError(
            f"{checkpoint_path} has no 'model_cfg' — it predates the JiT trainer "
            "and cannot be loaded by this pipeline (see archive/MANIFEST.md)."
        )
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
    if use_ema and ema_state:
        shadow = ema_state.get("shadow", {})
        applied = 0
        for name, param in model.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name].to(param.device))
                applied += 1
        print(f"  [model] EMA weights applied ({applied} tensors)")
    elif use_ema:
        print("  [model] WARNING: no EMA in checkpoint — using raw weights")

    model.eval().to(device)
    flow = FlowMatching()
    print(f"  [model] step {state.get('step')} | encoder {cfg.get('encoder', 'unrecorded')} "
          f"| h_dim {cfg['h_dim']} | hidden {cfg['hidden_dim']} | patch {cfg.get('patch_size')}")
    return model, flow, cfg


def load_probe_encoder(encoder_name: str, device: torch.device, cfg: dict | None = None,
                       checkpoint_path=None):
    """
    Load the frozen encoder and verify it matches the generator it will be used with.

    A generator's conditioning projector is built for one representation width.
    Pairing it with a different encoder produces plausible-looking but meaningless
    samples, so the h_dim check is an error, not a warning.

    Returns:
        (encoder, transform) — transform is always the 224 px ImageNet pipeline.
    """
    if cfg is not None and cfg.get("h_dim") != get_h_dim(encoder_name):
        recorded = cfg.get("encoder", "unrecorded")
        raise ValueError(
            f"Encoder '{encoder_name}' has h_dim {get_h_dim(encoder_name)}, but the "
            f"generator expects {cfg['h_dim']} (trained with encoder '{recorded}')."
        )
    kwargs = {"checkpoint_path": checkpoint_path} if checkpoint_path else {}
    encoder = get_encoder(encoder_name, device=str(device), **kwargs)
    return encoder, build_transform()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def unnorm(x: torch.Tensor) -> torch.Tensor:
    """Diffusion space [-1, 1] -> display space [0, 1]."""
    return (x.clamp(-1, 1) + 1.0) / 2.0


def set_plot_style() -> None:
    """One matplotlib style for every figure, so the report looks like one document."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    })


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(folder) -> list:
    """Sorted image paths directly inside ``folder`` (non-recursive)."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def stratified_sample(split_dir, n_per_class: int, seed: int = 0) -> list:
    """
    Sample n_per_class images from each class folder of a split.

    Stratification matters here: Messidor-2 is 58% no-DR, so a uniform sample
    would barely touch the advanced grades the probes are most interesting on.

    Returns:
        list of (class_name, Path) tuples, sorted by class.
    """
    import random

    rng = random.Random(seed)
    picked = []
    for class_dir in sorted(p for p in Path(split_dir).iterdir() if p.is_dir()):
        images = list_images(class_dir)
        if not images:
            print(f"  [warn] {class_dir.name}: no images, skipping")
            continue
        chosen = images if len(images) <= n_per_class else rng.sample(images, n_per_class)
        picked.extend((class_dir.name, p) for p in sorted(chosen))
    return picked
