"""
rcdm/conditioning.py

Conditioning utilities shared by the JiT denoiser.

Origin / changes vs upstream repos
------------------------------------
  RMSNorm              : NEW — not in RCDM or JiT as a standalone module.
                         JiT used nn.LayerNorm inside AdaLNZero; we factor it
                         out and replace every LayerNorm with RMSNorm.

  ConditionalBatchNorm2d : FROM RCDM (unchanged).
                           RCDM used this to condition the UNet backbone
                           (2-D spatial feature maps, one scale/bias per channel).
                           Kept for backward compatibility; not used by JiT path.

  ConditioningProjector  : FROM RCDM (adapted).
                           Original: Linear(2048 → 512) + SiLU — fixed dims for
                           ResNet-50 avgpool (2048-dim) output.
                           Changed: h_dim 2048 → 384 (DinoV3 ViT-S/16 CLS token);
                           output dim is now configurable via cond_dim (default 768).

  AdaLNZero              : NEW — from JiT / DiT (Peebles & Xie 2022).
                           RCDM conditioned a CNN via ConditionalBatchNorm2d.
                           JiT conditions a token sequence (B, N, D) via adaLN-Zero.
                           RCDM's cBN cannot be used here — it assumes spatial 2-D
                           feature maps (B, C, H, W), not token sequences.
                           Norm: nn.LayerNorm (JiT original) → RMSNorm (our change).
"""

import torch
import torch.nn as nn


# ── JiT-RCDM [fix-5a]: RMSNorm — replaces nn.LayerNorm in AdaLNZero and FinalLayer ──
# Why: RMSNorm drops the mean-centering step → faster, more stable in mixed precision.
# Standard in all modern ViT diffusion models (MAR, SiT, etc.) that follow JiT.
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich 2019).

    x_out = x / sqrt(mean(x²) + eps) * weight

    Differs from nn.LayerNorm in that mean-centering is omitted.
    affine=True  : learnable scale weight (used in qk-norm inside Attention)
    affine=False : no learnable parameters (used inside AdaLNZero where adaLN
                   already provides shift and scale)
    """

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm  = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x_out = x.float() * norm
        if self.weight is not None:
            x_out = x_out * self.weight
        return x_out.to(x.dtype)


# ── FROM RCDM — unchanged ──
# Used only by the legacy UNet path (guided_diffusion/). Not used by JiT.
class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization from RCDM.

    Conditions a 2-D CNN feature map (B, C, H, W) using a conditioning vector h.
    One learned scale γ and bias β per channel, derived from h via a linear layer.

    Kept for backward compatibility with the UNet path in guided_diffusion/.
    The JiT denoiser uses AdaLNZero instead (see below) because it operates on
    token sequences (B, N, D), not spatial feature maps.
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn       = nn.BatchNorm2d(num_features, affine=False, eps=1e-5, momentum=0.1)
        self.gamma_fc = nn.Linear(cond_dim, num_features)
        self.beta_fc  = nn.Linear(cond_dim, num_features)
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x_norm = self.bn(x)
        gamma  = self.gamma_fc(h).unsqueeze(-1).unsqueeze(-1)
        beta   = self.beta_fc(h).unsqueeze(-1).unsqueeze(-1)
        return gamma * x_norm + beta


# ── FROM RCDM (adapted) ──
# Original: Linear(2048 → 512) + SiLU — hard-coded for ResNet-50 avgpool (2048-dim).
# Changed:  h_dim parameter (2048 → 384 for DinoV3 ViT-S/16 CLS token);
#           cond_dim parameter (512 → configurable, default 768 for JiT-B).
#           JiT_S_16/S32 presets use cond_dim=128 as a regularising bottleneck.
class ConditioningProjector(nn.Module):
    """
    Projects the encoder CLS token h into the conditioning space.

        h (B, h_dim=384) → Linear(384 → cond_dim) + SiLU → h_proj (B, cond_dim)

    h_proj is then added to the timestep embedding to form the shared
    conditioning signal c that drives all adaLN-Zero blocks.

    The learned linear layer lets the model warp the frozen encoder's
    semantic space to align with the diffusion model's internal representation
    without modifying the encoder weights.

    Args:
        h_dim    : encoder CLS token dimension (384 for DinoV3 ViT-S/16)
        cond_dim : width of the shared conditioning signal c
                   (== hidden_dim → no bottleneck; < hidden_dim → bottleneck)
    """

    def __init__(self, h_dim: int = 384, cond_dim: int = 768):
        super().__init__()
        # ── changed from RCDM: h_dim 2048→384, cond_dim 512→configurable ──
        self.proj = nn.Sequential(
            nn.Linear(h_dim, cond_dim),
            nn.SiLU(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : (B, h_dim) → (B, cond_dim)"""
        return self.proj(h)


# ── NEW — from JiT / DiT (Peebles & Xie 2022) ──
# RCDM had no equivalent: RCDM conditioned a UNet via ConditionalBatchNorm2d
# which operates on 2-D spatial feature maps (B, C, H, W). AdaLNZero works on
# 1-D token sequences (B, N, D) — mandatory for a ViT backbone.
# Norm change vs JiT original: nn.LayerNorm → RMSNorm (fix-5a).
class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm Zero (adaLN-Zero) conditioning for transformer blocks.

    A per-block MLP maps the shared conditioning signal c to 6 modulation scalars:
        c → SiLU → Linear(cond_dim → 6·hidden_dim)
        → shift_a, scale_a, gate_a, shift_f, scale_f, gate_f

    Applied to the token sequence x:
        x ← x + gate_a · Attn( (1+scale_a) · RMSNorm(x) + shift_a )
        x ← x + gate_f · FFN(  (1+scale_f) · RMSNorm(x) + shift_f )

    Zero-init: the output projection is initialised to zero so all gates start
    at 0 → every block is an identity at training step 0. The model first
    learns the unconditional denoising trajectory; conditioning takes effect
    gradually as gates depart from zero. Same rationale as cBN initialising
    γ=1, β=0 in RCDM.

    Args:
        hidden_dim : ViT hidden dimension
        cond_dim   : dimension of the fused conditioning vector c
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        # ── JiT-RCDM [fix-5a]: RMSNorm replaces nn.LayerNorm ──
        self.norm1 = RMSNorm(hidden_dim, affine=False, eps=1e-6)
        self.norm2 = RMSNorm(hidden_dim, affine=False, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_dim),
        )
        # Zero-init: gates = 0 at step 0 → all blocks are identity at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def modulate(self, x, shift, scale):
        return (1 + scale.unsqueeze(1)) * x + shift.unsqueeze(1)

    def forward_pre(self, x: torch.Tensor, c: torch.Tensor):
        """
        Return the 6 modulation parameters from conditioning signal c.
        Called by JiTBlock which supplies its own Attention and FFN modules.

        Args:
            x : token sequence (B, N, hidden_dim)
            c : fused conditioning (B, cond_dim)

        Returns:
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f — each (B, hidden_dim)
        """
        return self.adaLN_modulation(c).chunk(6, dim=-1)
