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
| **Conditioning mechanism** | ConditionalBatchNorm2d | adaLN-Zero | adaLN-Zero (from JiT) |
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

## 3 — DinoV3 ViT-S/16: why and how

### Why DinoV3 over RCDM's ResNet-50

| Reason | Detail |
|---|---|
| **Domain specificity** | DinoV3 ViT-S/16 is fine-tuned on medical/retinal imagery; its CLS token is semantically aligned with Messidor-2 fundus images — vessel topology, disc morphology, pathology stage |
| **Perceptual similarity** | DINO-style SSL trains representations where nearby vectors correspond to perceptually similar images, without classification label pressure. ResNet-50 avgpool conflates classification-discriminative features with perceptual similarity |
| **Spatial awareness** | ViT attention can integrate long-range spatial relationships (vessel trees spanning the full image); CNN pyramids compress this away |
| **Compact dimension** | 384-dim (ViT-S) vs 2048-dim (ResNet-50) — keeps the conditioning path lightweight |
| **Flip invariance** | DinoV3 was trained with `RandomHorizontalFlip(p=0.5)` — the CLS token is flip-invariant, allowing free flip augmentation in `RepresentationDataset` |

### What changed in the encoder module

```python
# rcdm/encoder.py — ENTIRELY NEW FILE (not in RCDM or JiT)
DINOV3_CHECKPOINT = "checkpoints/dinov3_vits16_tmp"
ENCODER_OUTPUT_DIM = 384    # ViT-S/16 CLS token dimension

def load_encoder(device="cpu", checkpoint_path=DINOV3_CHECKPOINT):
    encoder = AutoModel.from_pretrained(checkpoint_path, local_files_only=True)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    return encoder.to(device)

def build_transform(image_size=224):
    # ALWAYS produces 224×224 — DinoV3 has a fixed 14×14 pos-embed grid (224/16=14).
    # Generator image_size is independent and does not affect this transform.
    return transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
```

### Local checkpoint requirement

The encoder is loaded from a local HuggingFace-format directory (`checkpoints/dinov3_vits16_tmp/`). `local_files_only=True` prevents any network request. The directory must contain:
- `config.json`
- `model.safetensors`

**Critical fix in `config.json`:** Set `"use_gated_mlp": false`. If this is `true`, HuggingFace will randomly initialise a missing `gate_proj` on every load, producing non-deterministic CLS tokens between runs. This is not a code change — edit the file manually before running `precompute_reps.py`.

### Precomputed representations

RCDM ran the encoder on every training step. We precompute all representations once:

```bash
python scripts/precompute_reps.py \
    --data_dir  data/messidor2/train \
    --out_file  data/messidor2/train_reps.pt \
    --batch_size 64 \
    --device cpu
```

Output: `{"paths": [...], "reps": Tensor(N, 384)}`. Index alignment is exact: `reps[i]` corresponds to `paths[i]`. The encoder never runs during the training loop.

---

## 4 — Soundness fixes

A cross-file audit identified five deviations from the published JiT/RCDM designs. Each fix is tagged in source code as `# ── JiT-RCDM [fix-N]: description ──`.

### Fix 1 — EMA not persisted in checkpoints `[fix-1]`

**Problem:** The EMA class existed and shadow weights were updated each step, but they were never written to the checkpoint dict. Resuming from a checkpoint silently discarded all EMA history. `sampling.py` also loaded raw model weights instead of the EMA shadow — defeating the purpose of EMA entirely.

**Fix:** Checkpoint now includes `"ema": ema.state_dict()`. `sampling.py` loads `ckpt["ema"]["shadow"]` and copies shadow weights into the model before sampling. Old checkpoints (no `"ema"` key) fall back to raw weights with a printed warning.

```python
# scripts/train.py
torch.save({"model": model.state_dict(), "ema": ema.state_dict(), ...}, path)

# scripts/sampling.py
shadow = ckpt["ema"]["shadow"]
model.load_state_dict(shadow, strict=False)   # strict=False: freqs_cis buffer not in EMA
```

### Fix 2 — DinoV3 encoder fed wrong image size in `sampling.py` `[fix-2]`

**Problem:** `sampling.py` called `build_transform(image_size=cfg["image_size"])`, where `image_size` is the *generator's* resolution (e.g. 256 or 512). DinoV3 has a fixed 14×14 positional grid (`224/16=14`); any other resolution silently corrupts the CLS token.

**Fix:** Hard-coded to 224 px everywhere the encoder transform is built.

```python
# ── JiT-RCDM [fix-2]: always 224 for DinoV3 ViT-S/16 ──
enc_transform = build_transform(image_size=224)
```

### Fix 3 — Null-h should be a learnable parameter `[fix-3]`

**Problem:** CFG null conditioning used `torch.zeros_like(h)`. A hard-coded zero vector is an arbitrary choice — it forces the unconditional branch to use an embedding that has no learned meaning.

**Fix:** `JiT.__init__` registers `self.null_h = nn.Parameter(torch.zeros(h_dim))`. It is part of the model's parameter set and is included in the EMA. During training, masked samples use `null_h` via `torch.where`. During inference, the unconditioned pass uses `model.null_h.expand(B, -1)`. The parameter converges to a representation of "no conditioning" rather than remaining a fixed zero.

### Fix 4 — JiT training recipe deviations `[fix-4a/b/c/d]`

Four hyperparameters that deviated from the published JiT recipe:

| Fix | Parameter | Was | Should be | Source |
|---|---|---|---|---|
| 4a | AdamW β₂ | 0.999 (PyTorch default) | **0.95** | JiT paper |
| 4b | t-sampler | logit-normal(0, 1) | **logit-normal(−0.8, 0.8)** | JiT Tab. 3 |
| 4c | LR schedule | Fixed LR | **Linear warmup (`--warmup_steps 1000`)** | JiT training details |
| 4d | Batch size | Effective=8 | `--grad_accum N` for simulated larger batch | JiT paper: batch 256–512 |

### Fix 5 — Architecture deviations from JiT `[fix-5a/b/c/d]`

Four architectural components that were not yet matching the JiT specification:

| Fix | Component | Was | Should be |
|---|---|---|---|
| 5a | Normalisation | `nn.LayerNorm` | `RMSNorm` (Zhang & Sennrich 2019) |
| 5b | Positional encoding | Learned absolute `pos_embed` | 2D RoPE (Su et al. 2021) |
| 5c | Attention | `nn.MultiheadAttention` (no qk-norm) | Custom MHA with Q/K-RMSNorm |
| 5d | FFN | Linear → GELU → Linear | SwiGLU, no bias (Shazeer 2020) |

**RMSNorm** drops mean-centering (`x / rms(x)` instead of `(x − μ) / std(x)`). Faster, more stable in mixed precision, used by all major ViT diffusion models following JiT.

**2D RoPE** replaces learned absolute positional embeddings. `PatchEmbed` no longer adds a `pos_embed`; instead, `compute_2d_rope_freqs(grid_size, head_dim)` precomputes a rotation buffer that `apply_rotary_emb` applies to Q and K inside every attention block. RoPE provides relative position encoding that generalises to image sizes not seen during training.

**qk-norm** applies `RMSNorm(head_dim, affine=True)` to the reshaped Q and K tensors before dot-product attention. Prevents attention logit divergence in early training.

**SwiGLU** replaces `Linear → GELU → Linear`. The gate and value projections are computed jointly, then element-wise multiplied after a SiLU activation. Inner dimension `= round_to_256(hidden_dim × mlp_ratio × 2/3)` maintains the same parameter count as a standard FFN.

---

## 5 — Files modified

| File | Origin | What changed |
|---|---|---|
| `rcdm/encoder.py` | **NEW** (not in RCDM or JiT) | DinoV3 ViT-S/16 loader; `build_transform` hardcoded to 224; `encode_batch` for precomputation |
| `rcdm/jit.py` | **NEW** (adapted from JiT `denoiser.py`) | ViT denoiser; `FlowMatching` (flow path, Heun sampler, CFG); adaLN-Zero; `class_emb` → `cond_proj(h)`; RMSNorm; 2D RoPE; SwiGLU; qk-norm; learnable `null_h`; presets `JiT_S_16` / `JiT_S_32` |
| `rcdm/conditioning.py` | FROM RCDM (`condition_helper.py`) | `RMSNorm` added; `ConditioningProjector` h_dim 2048→384, cond_dim configurable; `AdaLNZero` added (from JiT); `ConditionalBatchNorm2d` kept for UNet compat |
| `rcdm/dataset.py` | FROM RCDM (`dataset.py`) | h_dim comments 2048→384; image_size default 64→224; packed-format support (`{"images": uint8_tensor, "reps": tensor}`) added for Colab |
| `scripts/train.py` | FROM RCDM (`scripts/image_train.py`) | UNet+DDPM → JiT+FlowMatching; `--model S16/S32` flag; `--cfg_dropout`; EMA state saved in checkpoints; betas=(0.9,0.95); warmup; grad_accum |
| `scripts/sampling.py` | FROM RCDM (adapted) | EMA weights loaded at inference; encoder transform hard-coded to 224; `--cfg_scale` two-pass guidance |
| `scripts/precompute_reps.py` | FROM RCDM (adapted) | DinoV3 instead of ResNet-50; image_size hardcoded to 224 (not args.image_size); output shape (N,384); Messidor-2 defaults |
| `scripts/fid_eval.py` | **NEW** | FID evaluation: samples one generated image per test image, saves to `fid_singles/`, computes FID via pytorch-fid |
| `scripts/pack_dataset.py` | **NEW** | Packs images + reps into a single `.pt` file (`{"images": uint8, "reps": float32}`) for Colab deployment |
| `guided_diffusion/guided_diffusion/script_util.py` | FROM RCDM | h_dim default 2048→384 |

### Files unchanged (kept for backward compatibility)

| File | Note |
|---|---|
| `guided_diffusion/guided_diffusion/unet.py` | Legacy UNet + cBN; functional but unused by the JiT path |
| `guided_diffusion/guided_diffusion/gaussian_diffusion.py` | DDPM machinery retained; unused by the JiT training path |

---

## Data migration checklist

Precomputed representation files are **invalidated** when the encoder changes. Regenerate from scratch:

```bash
# 1. Edit checkpoints/dinov3_vits16_tmp/config.json → "use_gated_mlp": false

# 2. Precompute representations — output shape (N, 384)
python scripts/precompute_reps.py \
    --data_dir  data/messidor2/train \
    --out_file  data/messidor2/train_reps.pt \
    --device    cpu

# 3. Train with JiT_S_16 preset
python scripts/train.py \
    --model        S16 \
    --reps_file    data/messidor2/train_reps.pt \
    --save_dir     checkpoints/ \
    --image_size   224 \
    --cfg_dropout  0.1 \
    --total_steps  50000 \
    --save_interval 5000 \
    --log_interval  100 \
    --device       mps
```

Old `.pt` files with shape `(N, 2048)` will fail the assertion in `precompute_reps.py` and cause a shape mismatch in the model. They must be deleted and regenerated.

Checkpoints trained before fix-5 used `attn.in_proj_weight` (nn.MultiheadAttention) and had a `pos_embed` parameter. These are **incompatible** with the current architecture and must be retrained from scratch.
