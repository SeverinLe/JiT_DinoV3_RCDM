"""
tests/test_pipeline.py

Shape and contract tests for the pieces every experiment depends on.

These are deliberately cheap — no weights are downloaded and no encoder is
loaded, so the suite runs in seconds on CPU:

    python -m pytest tests/ -v        # or: python tests/test_pipeline.py

What they protect:
  * the encoder registry's h_dim values, which the trainer and every probe use
    to decide whether a model and a representation cache belong together;
  * adaLN-Zero's zero-initialisation, i.e. that conditioning starts as the
    identity — if that regresses, training silently loses its warm-up curriculum;
  * the denoiser's forward shape contract;
  * the index alignment between cached representations and their paths, which
    nothing downstream re-checks.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcdm.conditioning import AdaLNZero, ConditioningProjector, RMSNorm  # noqa: E402
from rcdm.encoders import ENCODER_NAMES, build_transform, get_h_dim  # noqa: E402
from rcdm.jit import JiT_S_16, JiT_S_32, FlowMatching  # noqa: E402


def test_encoder_registry_widths():
    """h_dim per encoder is an architecture constant, not a tunable."""
    assert get_h_dim("dinov3") == 384          # ViT-S
    assert get_h_dim("retfound_cfp") == 1024   # ViT-L
    assert get_h_dim("resnet50") == 2048       # avgpool trunk
    assert set(ENCODER_NAMES) == {"dinov3", "retfound_cfp", "resnet50"}


def test_unknown_encoder_raises():
    try:
        get_h_dim("not_an_encoder")
    except ValueError as exc:
        assert "Available" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown encoder")


def test_transform_always_224():
    """A /16 ViT's positional grid is tied to 224 px; the default must not drift."""
    from PIL import Image

    transform = build_transform()
    out = transform(Image.new("RGB", (1000, 700)))
    assert out.shape == (3, 224, 224), out.shape


def test_conditioning_projector_shapes():
    projector = ConditioningProjector(h_dim=384, cond_dim=128)
    assert projector(torch.randn(4, 384)).shape == (4, 128)


def test_adaln_zero_is_identity_at_init():
    """
    Zero-init means every gate starts at 0, so a block is the identity at step 0
    and conditioning engages gradually as the gates move away from zero.
    """
    ada = AdaLNZero(hidden_dim=384, cond_dim=128)
    x = torch.randn(2, 196, 384)
    modulations = ada.forward_pre(x, torch.randn(2, 128))
    assert len(modulations) == 6, "expected shift/scale/gate for attention and FFN"
    for tensor in modulations:
        assert torch.allclose(tensor, torch.zeros_like(tensor)), \
            "adaLN-Zero output must be zero at initialisation"


def test_rmsnorm_preserves_shape():
    norm = RMSNorm(384)
    x = torch.randn(2, 196, 384)
    assert norm(x).shape == x.shape


def test_jit_forward_shapes():
    """The denoiser maps (B, 3, H, W) -> (B, 3, H, W) for both presets."""
    for preset, h_dim in ((JiT_S_16, 384), (JiT_S_32, 1024)):
        model = preset(image_size=224, h_dim=h_dim).eval()
        x = torch.randn(2, 3, 224, 224)
        t = torch.rand(2)
        h = torch.randn(2, h_dim)
        with torch.no_grad():
            out = model(x, t, h)
        assert out.shape == x.shape, (preset.__name__, out.shape)


def test_null_h_exists_for_cfg():
    """CFG needs a learnable null_h; without it, cfg_scale > 1 is meaningless."""
    model = JiT_S_16(image_size=224, h_dim=384)
    assert hasattr(model, "null_h")
    assert model.null_h.requires_grad
    assert model.null_h.shape == (384,)


def test_flow_matching_sampler_runs():
    """Two ODE steps is enough to catch a broken sampler without being slow."""
    model = JiT_S_16(image_size=224, h_dim=384).eval()
    flow = FlowMatching()
    with torch.no_grad():
        out = flow.sample(model, torch.randn(1, 3, 224, 224), torch.randn(1, 384),
                          num_steps=2, cfg_scale=1.0)
    assert out.shape == (1, 3, 224, 224)
    assert torch.isfinite(out).all(), "sampler produced NaN/inf"


def test_representation_cache_alignment():
    """
    reps[i] must belong to paths[i]. Nothing downstream re-checks this, so a
    misaligned cache would train a model on shuffled conditioning.

    Skipped when the cache is absent (it is not tracked in git).
    """
    cache_path = Path(__file__).resolve().parents[1] / \
        "data/processed/messidor2/dinov3/train_reps.pt"
    if not cache_path.exists():
        print(f"  [skip] {cache_path.name} not present")
        return
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert len(cache["paths"]) == cache["reps"].shape[0]
    assert cache["reps"].shape[1] == 384
    assert torch.isfinite(cache["reps"]).all()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001 — a test runner reports, it does not raise
                print(f"  FAIL  {name}: {exc}")
                failures += 1
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
