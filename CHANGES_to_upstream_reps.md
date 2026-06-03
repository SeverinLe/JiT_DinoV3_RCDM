# JiT-RCDM — Change Log

This project combines two upstream repositories:

- **[facebookresearch/RCDM](https://github.com/facebookresearch/RCDM)** — representation-conditioned diffusion with a frozen SSL encoder and a UNet + DDPM backbone.
- **[LTH14/JiT](https://github.com/LTH14/JiT)** — ViT-based diffusion model using adaLN-Zero conditioning, flow matching, and the Heun ODE sampler.

We replace RCDM's UNet+DDPM backbone with JiT's ViT, swap RCDM's ResNet-50 encoder for a domain-specific **DinoV3 ViT-S/16**, and introduce several soundness fixes to match the published designs of both repos.

Every deviation from the originals is documented here.

---

## Final state at a glance

| Component | Original RCDM | Original JiT | JiT-RCDM (this repo) |
|---|---|---|---|
| **Encoder** | ResNet-50, 2048-dim avgpool | *(no image encoder — class labels)* | DinoV3 ViT-S/16, 384-dim CLS token |
| **Conditioning input** | SSL repr h (continuous) | Integer class label y | SSL repr h (continuous) from DinoV3 |
| **Conditioning projector** | Linear(2048→512) + SiLU | `class_emb` lookup table | Linear(384→cond_dim) + SiLU |
| **Conditioning dim** | 512 | hidden_dim | `cond_dim` — 128 in S presets (regularising bottleneck) |
| **Conditioning mechanism** | Added to emb | adaLN-Zero | adaLN-Zero (from JiT) |
| **Denoiser** | UNet | ViT (JiT) | ViT (JiT) |
| **Normalisation** | BatchNorm / LayerNorm | RMSNorm | RMSNorm (from JiT) |
| **Positional encoding** | Learned absolute | 2D RoPE | 2D RoPE (from JiT) |
| **FFN** | Linear → GELU → Linear | SwiGLU, no bias | SwiGLU, no bias (from JiT) |
| **Attention** | `nn.MultiheadAttention` | Custom MHA + qk-norm | Custom MHA + qk-norm + RoPE (from JiT) |
| **Prediction target** | ε (noise) | x (clean image) | x (from JiT) |
| **Noise schedule** | DDPM cosine/linear | Linear flow, logit-normal(−0.8, 0.8) t | Linear flow (from JiT) |
| **Sampler** | DDPM 1000 steps | 50-step Heun ODE | Heun ODE (from JiT) |
| **Null conditioning** | `zeros_like(h)` | Learned `null_class` embedding | Learnable `nn.Parameter` `null_h` |
| **EMA** | Not used | Saved; applied at inference | Saved in checkpoints; applied at inference |
| **AdamW β₂** | 0.999 (default) | 0.95 | 0.95 (from JiT) |
| **LR schedule** | Fixed LR | Linear warmup + constant | Linear warmup (from JiT) |
| **Preset variants** | — | — | `JiT_S_16` (~25 M), `JiT_S_32` (~26 M) |

---

## 1 — What we took from RCDM and what we changed

### Taken from RCDM (unchanged or lightly adapted)

| Component | RCDM file | Status here |
|---|---|---|
| `RepresentationDataset` concept | `dataset.py` | `rcdm/dataset.py` — loads `(image, h)` pairs; `image_size` default updated to 224; packed-format support added |
| `ConditioningProjector` structure | `guided_diffusion/condition_helper.py` | `rcdm/conditioning.py` — `Linear(h_dim, cond_dim) + SiLU`; h_dim and cond_dim made configurable |
| `ConditionalBatchNorm2d` | `guided_diffusion/condition_helper.py` | Kept in `rcdm/conditioning.py` for backward compat with the UNet path; **not used in the JiT training path** |
| Classifier-free guidance concept (null-h dropout) | `scripts/image_train.py` | `FlowMatching.training_loss` — `p_uncond` fraction replaces `h` with `null_h` |
| Dual normalisation (encoder ImageNet vs diffusion [−1,1]) | Implicit in RCDM pipeline | `rcdm/dataset.py` + `rcdm/encoder.py` — kept separate, never mixed |
| Frozen encoder (no gradients) | `guided_diffusion/condition_helper.py` | `rcdm/encoder.py` — `encoder.eval(); requires_grad=False` |

### Changed from RCDM

#### Encoder: ResNet-50 → DinoV3 ViT-S/16

RCDM's encoder was `torchvision.models.resnet50` with the final FC layer removed, producing a 2048-dim avgpool vector. We replace this entirely with DinoV3 ViT-S/16 (see §3).

```python
# RCDM
encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
encoder.fc = nn.Identity()
h = encoder(x)                              # (B, 2048)

# JiT-RCDM
encoder = AutoModel.from_pretrained("checkpoints/dinov3_vits16_tmp", local_files_only=True)
h = encoder(pixel_values=x).last_hidden_state[:, 0, :]   # CLS token → (B, 384)
```

#### Conditioning projector: h_dim 2048→384, output configurable

```python
# RCDM
ConditioningProjector(h_dim=2048, cond_dim=512)   # fixed dims

# JiT-RCDM
ConditioningProjector(h_dim=384, cond_dim=128)    # h_dim follows encoder; cond_dim is a preset choice
```

The output dimension `cond_dim=128` is our own design choice — not from RCDM (512) and not from JiT (which uses `hidden_dim`). It acts as a regularising bottleneck for the small Messidor-2 dataset, reducing the adaLN-Zero MLP size from `hidden_dim→6·hidden_dim` to `128→6·hidden_dim` per block.

#### ConditionalBatchNorm2d → AdaLNZero

RCDM conditioned a UNet via `ConditionalBatchNorm2d`, which modulates 2-D spatial feature maps `(B, C, H, W)`. The JiT denoiser operates on token sequences `(B, N, D)` — cBN has no meaning there.

We replace cBN with `AdaLNZero` (from DiT / JiT): per-block, a shared conditioning vector `c` produces 6 modulation scalars (shift + scale + gate for attention and FFN). The adaLN output is zero-initialised so all gates start at 0 → every block is an identity at step 0. Conditioning takes effect gradually as gates depart from zero. This is the same curriculum rationale as cBN's `γ=1, β=0` initialisation in RCDM.

```python
# RCDM — 2-D spatial modulation
out = self.cbn(feature_map, h)   # (B, C, H, W)

# JiT-RCDM — token sequence modulation (6 scalars per block)
shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaLN_modulation(c).chunk(6, dim=-1)
x = x + gate_a * Attn((1 + scale_a) * RMSNorm(x) + shift_a)
x = x + gate_f * FFN( (1 + scale_f) * RMSNorm(x) + shift_f)
```

#### Prediction target and sampler

RCDM uses ε-prediction (predict added noise) over a 1000-step DDPM chain with a cosine/linear β schedule. We use x-prediction (predict clean image) over a linear flow path, sampled with a 50-step Heun ODE — this comes from JiT (see §2).

---

## 2 — What we took from JiT and what we changed

### Taken from JiT (unchanged or lightly adapted)

| Component | JiT file | Status here |
|---|---|---|
| `JiT` ViT denoiser class | `denoiser.py` | `rcdm/jit.py` — block structure, patch embed, final layer |
| `AdaLNZero` conditioning | `denoiser.py` | `rcdm/conditioning.py` + `rcdm/jit.py` — 6-param modulation, zero-init output |
| Sinusoidal timestep embedding | `denoiser.py` | `rcdm/jit.py` — `timestep_embedding()` |
| `time_embed` MLP (Linear→SiLU→Linear) | `denoiser.py` | `rcdm/jit.py` — width `cond_dim → 4·cond_dim → cond_dim` |
| Heun ODE sampler | `sample.py` | `rcdm/jit.py` — `FlowMatching.sample`; 50 steps; pure Euler at last step |
| CFG two-pass x-pred blending | `sample.py` | `rcdm/jit.py` — blending at `x_pred` level (not velocity level) |
| EMA shadow-weight pattern | `denoiser.py` | `scripts/train.py` — `EMA` class; `apply_shadow` / `restore` |
| 2D RoPE | `denoiser.py` | `rcdm/jit.py` — `compute_2d_rope_freqs`; applied to Q/K in every block |
| SwiGLU FFN, no bias | `denoiser.py` | `rcdm/jit.py` — `SwiGLU` module |
| qk-norm | `denoiser.py` | `rcdm/jit.py` — `RMSNorm` applied to Q and K per head |
| RMSNorm | `denoiser.py` | `rcdm/conditioning.py` — `RMSNorm` class |
| AdamW β₂=0.95 | JiT paper | `scripts/train.py` |
| logit-normal(−0.8, 0.8) t-sampler | JiT paper Tab. 3 | `rcdm/jit.py` — `FlowMatching.training_loss` |

### Changed from JiT

#### Class label y → continuous SSL repr h

JiT is class-conditional: it looks up an integer label `y` in an embedding table and adds it to the timestep embedding. Messidor-2 has only 5 severity grades — mapping all images within a grade to the same point would lose fine-grained variation. We replace the label embedding with `ConditioningProjector(h)`:

```python
# Original JiT
c = timestep_emb(t) + class_emb(y)     # y is an integer class label

# JiT-RCDM
c = timestep_emb(t) + cond_proj(h)     # h is a continuous 384-dim CLS token
```

There is no `class_emb` table anywhere in `rcdm/jit.py`. The substitution is complete.

#### Conditioning dimension: hidden_dim → cond_dim=128

JiT's conditioning path lives entirely at `hidden_dim` (768 for JiT-B): both the timestep embedding and the class embedding are `hidden_dim`-wide, and the adaLN MLP maps `hidden_dim → 6·hidden_dim`. We decouple this with a separate `cond_dim` parameter and set `cond_dim=128` in our S presets. This reduces each adaLN MLP from `768→4608` to `128→4608`, cutting conditioning-path parameters ~6× for the small training budget.

#### CFG null conditioning: `zeros_like(h)` → learnable `null_h`

JiT uses a learned `null_class` embedding for CFG. Our h is continuous, not a class index, so we register a learnable parameter on the model:

```python
# JiT-RCDM [fix-3]: learnable null-h — converges to the representation centroid
self.null_h = nn.Parameter(torch.zeros(h_dim))
```

During training, `p_uncond` fraction of examples substitute `null_h` for `h`. During inference, `null_h` is used for the unconditioned pass. Because `null_h` is part of the model's parameter set (and included in the EMA), it converges to a learned representation rather than a fixed zero vector.

#### Presets: JiT_S_16 and JiT_S_32

JiT ships a single JiT-B/16 configuration (hidden_dim=768, patch=16). We add two smaller presets for local/MPS training:

| Preset | `hidden_dim` | `num_heads` | `patch_size` | `cond_dim` | Tokens @ 224px | Params |
|---|---|---|---|---|---|---|
| `JiT_S_16` | 384 | 6 | 16 | 128 | 196 | ~25 M |
| `JiT_S_32` | 384 | 6 | 32 | 128 | 49 | ~26 M |

`patch_size=16` is the recommended default — 16 px/patch preserves fine retinal structures (micro-aneurysms, thin vessels). `patch_size=32` is kept for fast local experiments where memory is constrained.

---

