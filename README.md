# JiT-RCDM — Representation-Conditioned Diffusion for Retinal Images

A research implementation that merges two open-source frameworks:

- **RCDM** ([facebookresearch/RCDM](https://github.com/facebookresearch/RCDM)) — the idea of conditioning a diffusion model on an SSL representation `h` instead of a class label
- **JiT** ([LTH14/JiT](https://github.com/LTH14/JiT)) — a plain ViT denoiser trained with flow matching

Applied to **Messidor-2** diabetic retinopathy fundus images, conditioned on a frozen **DinoV3 ViT-S/16** encoder (384-dim CLS token).

---

## Architecture overview

```
 Conditioning image (224×224, any fundus image)
        │
        ▼  build_transform(224): Resize + CenterCrop + ImageNet normalise
 ┌──────────────────────────────┐
 │  DinoV3 ViT-S/16  [frozen]  │   CLS token → h  (B, 384)
 └──────────────────────────────┘
        │
        ▼  ConditioningProjector: Linear(384 → 128) + SiLU
        │
        h_proj  (B, 128)
        │
        │  +  time_embed MLP: sinusoidal(t×1000, 128) → Linear → SiLU → Linear
        │     t_emb  (B, 128)
        │
        └────  c = h_proj + t_emb   (B, 128)    ← shared conditioning signal
                        │
                        ▼  (broadcast to every transformer block)
        ┌──────────────────────────────────────────┐
        │  JiT_S_16 ViT Denoiser                   │
        │                                          │
        │  PatchEmbed(patch=16)                    │
        │    z_t (B, 3, 224, 224)                  │
        │    → tokens  (B, 196, 384)               │
        │                                          │
        │  × 12  JiTBlock                          │
        │    adaLN-Zero(c):  128 → 6×384           │
        │    Attention  (RoPE + qk-norm)           │
        │    SwiGLU FFN                            │
        │                                          │
        │  FinalLayer → unpatchify                 │
        │    → x_pred  (B, 3, 224, 224)            │
        └──────────────────────────────────────────┘
```

**Model preset — `JiT_S_16`:**

| Parameter | Value |
|---|---|
| `hidden_dim` | 384 |
| `depth` | 12 |
| `num_heads` | 6 (head_dim = 64) |
| `patch_size` | 16 → 196 tokens at 224px |
| `cond_dim` | 128 (conditioning bottleneck) |
| `h_dim` | 384 (DinoV3 CLS token) |
| Parameters | ~25 M |

**Training objective (flow matching, x-prediction):**
```
z_t  = t·x + (1−t)·ε         t ~ sigmoid(N(−0.8, 0.8))
loss = MSE( JiT(z_t, t, h) , x )
```

**Inference (50-step Heun ODE with CFG):**
```
z₀ ~ N(0, I)
for each step:
    x_cond   = model(z, t, h)
    x_uncond = model(z, t, null_h)         ← learned null parameter
    x_pred   = x_uncond + cfg_scale · (x_cond − x_uncond)
    v = (x_pred − z) / (1 − t)            ← ODE velocity
    Heun correction step
→ z₅₀ ≈ generated image conditioned on h
```

---

## Conditioning pipeline — how `h` drives generation

This section explains the full path from a conditioning fundus image to the modulation of every transformer block. Understanding the dimensions at each step is important for reproducing or extending the work.

### Step 1 — Encode the conditioning image

```
Conditioning image (any resolution, fundus photo)
    ↓  Resize(224) + CenterCrop(224) + ImageNet normalise
    ↓  DinoV3 ViT-S/16  [frozen — never updated during training]
    ↓  last_hidden_state[:, 0, :]   ← CLS token only (index 0)
h : (B, 384)
```

**Why 384?** DinoV3 ViT-S/16 has `hidden_dim=384` — every token in the transformer, including CLS, is 384-dimensional. This is an architecture constant of ViT-S; ViT-B gives 768, ViT-L gives 1024. We use ViT-S because it is the smallest domain-fine-tuned model available, keeping the conditioning path lightweight.

**Why the CLS token?** The CLS token attends to all 196 patch tokens throughout all 12 encoder layers. By the final layer it aggregates global semantic content — pathology stage, disc location, vessel density — without being tied to any particular spatial position. Patch tokens (indices 5–200, after 4 register tokens at 1–4) carry local spatial information; they could be used for spatially-conditioned generation but are not used here.

**Why 224×224 for the encoder?** DinoV3 ViT-S/16 has a fixed positional embedding grid of 14×14 patches (224 ÷ 16 = 14). Any other input resolution would silently interpolate or corrupt these positional embeddings. The encoder always runs at 224px regardless of what resolution the generative model produces.

---

### Step 2 — Project h into the conditioning space

```
h : (B, 384)
    ↓  Linear(384 → 128)
    ↓  SiLU
h_proj : (B, 128)
```

**Why a projection at all?** The frozen encoder's CLS token lives in a representation space optimised for image similarity — not for steering a diffusion denoiser. The learned `Linear + SiLU` lets the denoiser warp that space to align with its internal activations, without touching the frozen encoder.

**Why output dimension 128 (`cond_dim=128`)?** This is a regularising bottleneck. The full 384-dim CLS token contains redundant information for our small 972-image dataset. Compressing to 128 before adding to the timestep signal forces the model to keep only the dominant axes of variation in `h`. It also reduces the parameter count of the adaLN MLPs in every block (each goes from `cond_dim → 6×hidden_dim`; smaller `cond_dim` = smaller adaLN MLP). Setting `cond_dim=hidden_dim=384` (no bottleneck) is the paper-faithful JiT setting and can be used for larger datasets.

**How this differs from RCDM:**
RCDM used `Linear(2048 → 512) + SiLU` — the same one-layer design. We kept the design but changed `2048 → 384` (encoder switch) and `512 → 128` (smaller bottleneck for smaller dataset).

---

### Step 3 — Fuse with the timestep embedding

```
t : (B,)   scalar in [0, 1]
    ↓  sinusoidal(t × 1000, 128)      ← scale to [0,1000] to match DDPM freq range
    ↓  Linear(128 → 512) → SiLU → Linear(512 → 128)
t_emb : (B, 128)

c = h_proj + t_emb    (B, 128)        ← single vector, computed once per forward pass
```

`c` fuses both signals — *what* the image looks like (from `h`) and *how much noise* is present (from `t`). The model cannot separate them — they are added before the first block and stay combined throughout. This means at every layer the model jointly uses both signals when deciding how to update each patch token.

**Why add instead of concatenate?** Concatenation would double `c` to 256-dim, increasing adaLN MLP parameters by 2× in all 12 blocks. Addition keeps `c` at 128-dim and naturally allows the timestep signal to dominate early in training (when `h_proj` weights are near-zero from random init), providing a natural curriculum: denoise first, steer with `h` later.

---

### Step 4 — adaLN-Zero: conditioning inside every block

`c` is broadcast to all 12 transformer blocks. Each block has its own small adaLN MLP:

```
c : (B, 128)
    ↓  SiLU → Linear(128 → 6×384)
    ↓  chunk into 6 vectors, each (B, 384)
shift_a, scale_a, gate_a   ← for the attention branch
shift_f, scale_f, gate_f   ← for the FFN branch

x ← x + gate_a · Attention( (1 + scale_a) · RMSNorm(x) + shift_a )
x ← x + gate_f · SwiGLU(   (1 + scale_f) · RMSNorm(x) + shift_f )
```

This happens 12 times (one per block) plus once in the final layer. Conditioning is thus applied **25 times** per forward pass. Every single attention and FFN computation is modulated by both `h` and `t` simultaneously.

**Zero-init:** The `Linear(128 → 6×384)` output projection is initialised to zero. At training step 0, all gates are 0 → every block is an identity function. The model stabilises in unconditional mode first; conditioning gradually engages as the gates depart from zero.

**Why not Conditional Batch Norm (RCDM's approach)?** RCDM's cBN operates on 2-D spatial feature maps `(B, C, H, W)` — one scale/bias per channel. A ViT operates on 1-D token sequences `(B, N, D)`. cBN cannot be applied here; adaLN is the standard solution for token sequences.

---

### Step 5 — CFG: null_h for unconditional generation

```python
self.null_h = nn.Parameter(torch.zeros(384))   # trainable, updated alongside the model
```

During training, 10% of batch elements have their `h` replaced by `null_h` (CFG dropout). During inference, the model runs twice per step:

```
x_pred = x_uncond + cfg_scale × (x_cond − x_uncond)
```

`null_h` is a **learnable** parameter — it converges toward the centroid of the representation space (the "average" retinal image direction) over training. RCDM used a fixed `torch.zeros_like(h)` as the null vector; zero is an arbitrary point in representation space. The learnable version is taken directly from JiT's `null_class` embedding for class-conditional generation.

**Critical fix:** the original implementation had `null.detach()` which blocked all gradients to `null_h` — it never actually updated and was identical to RCDM's hard-coded zeros. Removing `.detach()` allows gradients to flow normally.

---

### Dimension summary

| Stage | Tensor | Shape | Notes |
|---|---|---|---|
| Encoder output | `h` | `(B, 384)` | DinoV3 ViT-S/16 CLS token |
| After projection | `h_proj` | `(B, 128)` | `cond_dim=128` bottleneck |
| Timestep embedding | `t_emb` | `(B, 128)` | sinusoidal → 2-layer MLP |
| Fused conditioning | `c` | `(B, 128)` | `h_proj + t_emb` |
| adaLN output | 6 × modulation | `6 × (B, 384)` | per block, per branch |
| Patch tokens | `x` | `(B, 196, 384)` | 14×14 grid, hidden_dim=384 |
| Null conditioning | `null_h` | `(384,)` | learnable, for CFG |

---

## Source file attribution

This section documents exactly which files were taken from each upstream repo, and what was changed to make the pipeline work.

### Files from RCDM (`facebookresearch/RCDM`)

| File | Status | Changes made |
|---|---|---|
| `guided_diffusion/` | Kept unchanged | Legacy ADM UNet + DDPM — kept for reference only, not used by JiT path |
| `rcdm/dataset.py` | Adapted | (1) `h_dim` 2048→384; (2) added dual-format detection (standard paths vs packed uint8 images); (3) normalisation `[0,1]→[−1,1]` path for packed tensors |
| `rcdm/conditioning.py` | Adapted | (1) `ConditioningProjector` input dim 2048→384, output dim configurable via `cond_dim`; (2) added `RMSNorm`; (3) added `AdaLNZero` (replaces `ConditionalBatchNorm2d` for token sequences) |
| `scripts/precompute_reps.py` | Adapted | (1) Encoder ResNet-50→DinoV3; (2) rep dim 2048→384; (3) `build_transform` hardcoded to 224 (encoder fixed grid — not tied to `--image_size`) |
| `scripts/train.py` | Adapted | (1) Added EMA checkpoint persistence (was missing); (2) AdamW β₂ corrected to 0.95; (3) linear warmup; (4) gradient accumulation; (5) CFG dropout; (6) preset-based model construction |

### Files from JiT (`LTH14/JiT`)

| File | Status | Changes made |
|---|---|---|
| `rcdm/jit.py` | Adapted | (1) `class_emb(y)` integer label → `cond_proj(h)` continuous vector; (2) `null_class` int embedding → learnable `null_h: nn.Parameter(384,)`; (3) removed `.detach()` from null-h branch (was blocking gradients); (4) `nn.LayerNorm` → `RMSNorm` throughout; (5) `nn.MultiheadAttention` → custom `Attention` with qk-norm + 2D RoPE; (6) GELU FFN → `SwiGLU`; (7) learned `pos_embed` removed (RoPE replaces it); (8) logit-normal t-sampler μ corrected to −0.8; (9) `JiT_S_16` / `JiT_S_32` factory presets added |

### New files (written for JiT-RCDM, not present in either upstream repo)

| File | Purpose |
|---|---|
| `rcdm/encoder.py` | Load and freeze DinoV3 ViT-S/16 from local HuggingFace checkpoint; `build_transform(224)`; `encode_batch()` |
| `scripts/sampling.py` | Inference script: load EMA checkpoint, run 50-step Heun ODE, save image grids |
| `scripts/pack_dataset.py` | Convert `train_reps.pt` (paths + reps) → `train_packed.pt` (uint8 images + reps) for self-contained Colab training |
| `colab_training.ipynb` | End-to-end Google Colab A100 notebook: clone repo, copy data from Drive, train, sample |

### Critical fixes required to make the pipeline work

These are not optional improvements — without them the model produces wrong or non-deterministic results:

| Fix | File | Problem | Solution |
|---|---|---|---|
| `use_gated_mlp: false` | `checkpoints/dinov3_vits16_tmp/config.json` | Config declared 3-proj gated FFN; checkpoint only has 2-proj FFN. `gate_proj` was randomly re-initialised on every load → `h` vectors non-deterministic between training and inference | Set `"use_gated_mlp": false` in `config.json`; recompute `train_reps.pt` |
| Remove `null_h.detach()` | `rcdm/jit.py` — `FlowMatching.training_loss` | `.detach()` blocked all gradients to `null_h`; the learnable null parameter never updated | Removed `.detach()` — gradients flow normally |
| EMA checkpoint persistence | `scripts/train.py` | EMA shadow weights were updated during training but never saved to disk; resuming discarded all EMA history | Added `"ema": ema.state_dict()` to both periodic and final checkpoint saves |
| Encoder transform fixed to 224 | `scripts/precompute_reps.py` | `build_transform(args.image_size)` — if `--image_size` ≠ 224, encoder received wrong resolution (DinoV3 has fixed 14×14 positional grid) | Changed to `build_transform(image_size=224)` unconditionally |

---

## Prerequisites

| Requirement | Tested version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.1+ |
| torchvision | 0.16+ |
| Hugging Face `transformers` | 4.38+ |
| `safetensors` | 0.4+ |
| Pillow | 10+ |
| tqdm | any |

---

## Installation

```bash
# 1. Clone the branch
git clone --branch claude/silly-faraday-d8512b <repo_url>
cd master_implementation

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers safetensors pillow tqdm

# 4. (Optional) Install guided_diffusion for legacy UNet reference
pip install -e guided_diffusion/
```

---

## Project layout

```
master_implementation/
├── checkpoints/
│   └── dinov3_vits16_tmp/        ← DinoV3 encoder checkpoint (HuggingFace format)
│       ├── config.json           ← must have "use_gated_mlp": false
│       └── model.safetensors
│
├── data/
│   └── messidor2/
│       ├── train/                ← training images (flat or nested layout)
│       ├── test/                 ← conditioning images for sampling
│       ├── train_reps.pt         ← produced by precompute_reps.py
│       └── train_packed.pt       ← produced by pack_dataset.py (Colab use)
│
├── rcdm/
│   ├── encoder.py                ← DinoV3 loader + encode_batch()          [new]
│   ├── conditioning.py           ← RMSNorm, ConditioningProjector, AdaLNZero  [RCDM adapted]
│   ├── jit.py                    ← JiT_S_16 denoiser + FlowMatching         [JiT adapted]
│   └── dataset.py                ← RepresentationDataset (standard + packed) [RCDM adapted]
│
├── scripts/
│   ├── precompute_reps.py        ← step 1: encode all training images        [RCDM adapted]
│   ├── pack_dataset.py           ← step 1b: pack images for Colab            [new]
│   ├── train.py                  ← step 2: train JiT-RCDM                   [RCDM+JiT adapted]
│   └── sampling.py               ← step 3: generate images                  [new]
│
├── guided_diffusion/             ← legacy ADM UNet + DDPM (reference only)  [RCDM unchanged]
│
├── colab_training.ipynb          ← Google Colab A100 training notebook       [new]
├── PIPELINE.md                   ← full architecture + design decisions
├── CHANGES.md                    ← complete change log vs upstream repos
└── README.md                     ← this file
```

---

## Step 0 — Place the DinoV3 checkpoint

The encoder loads **only from local files**. Place the HuggingFace checkpoint at:

```
checkpoints/dinov3_vits16_tmp/
    config.json           ← must contain "use_gated_mlp": false
    model.safetensors     (or pytorch_model.bin)
```

**Verify `config.json` is correct:**
```bash
python -c "
import json
cfg = json.load(open('checkpoints/dinov3_vits16_tmp/config.json'))
assert not cfg.get('use_gated_mlp', False), 'Fix: set use_gated_mlp to false'
print('config OK — use_gated_mlp:', cfg.get('use_gated_mlp'))
"
```

**Verify the encoder loads and produces 384-dim vectors:**
```bash
python -c "
from rcdm.encoder import load_encoder, build_transform
enc = load_encoder(device='cpu')
print('Params:', sum(p.numel() for p in enc.parameters()))
# Expected: ~22M
"
```

---

## Step 1 — Precompute DinoV3 representations

Run the frozen encoder over every training image **once** and save a `(N, 384)` tensor. Training reads `h` directly from this file — the encoder never runs during training.

```bash
python scripts/precompute_reps.py \
    --data_dir   data/messidor2/train \
    --out_file   data/messidor2/train_reps.pt \
    --batch_size 64 \
    --device     cuda
```

> The encoder **always** processes images at 224×224, regardless of any other `--image_size` argument. DinoV3 ViT-S/16 has a fixed 14×14 positional grid (224 / 16 = 14).

Expected output:
```
[1/3] Collecting image paths...
Found 972 images in data/messidor2/train

[2/3] Loading encoder on cuda...
Running encoder over 972 images (batch_size=64)...

Representations shape : torch.Size([972, 384])
Representations dtype : torch.float32
Sample norm (first 5) : [14.2, 13.8, 15.1, 14.6, 13.9]

[3/3] Saving to data/messidor2/train_reps.pt...
Done. Saved 972 representations to data/messidor2/train_reps.pt
File size: 1.5 MB
Verification passed — paths and reps are aligned.
```

> **If you change `config.json`, the encoder weights, or the image preprocessing — delete `train_reps.pt` and rerun this step.** Stale representations produce silently wrong conditioning vectors.

---

## Step 1b — Pack the dataset (required for Colab)

`train_reps.pt` stores absolute paths from your local machine. On any other machine those paths don't exist. The packed format embeds the actual pixel data in the `.pt` file:

```bash
python scripts/pack_dataset.py \
    --reps_file  data/messidor2/train_reps.pt \
    --out_file   data/messidor2/train_packed.pt \
    --image_size 224
# Output: ~150 MB self-contained file
```

Upload `train_packed.pt` and the `dinov3_vits16_tmp/` folder to Google Drive before running the Colab notebook.

---

## Step 2 — Train

### Local (MPS / CUDA)

```bash
python scripts/train.py \
    --model         S16 \
    --reps_file     data/messidor2/train_reps.pt \
    --save_dir      checkpoints/ \
    --image_size    224 \
    --batch_size    8 \
    --grad_accum    4 \
    --lr            1e-4 \
    --warmup_steps  1000 \
    --cfg_dropout   0.1 \
    --total_steps   100000 \
    --save_interval 5000 \
    --log_interval  100 \
    --device        mps
```

Effective batch size = `batch_size × grad_accum` = **32**.

### Google Colab A100

Use `colab_training.ipynb` or run:

```bash
python scripts/train.py \
    --model         S16 \
    --reps_file     /content/train_packed.pt \
    --save_dir      checkpoints/ \
    --image_size    224 \
    --batch_size    128 \
    --grad_accum    1 \
    --lr            1e-4 \
    --warmup_steps  2000 \
    --cfg_dropout   0.1 \
    --total_steps   50000 \
    --save_interval 5000 \
    --wandb_project jit-rcdm \
    --device        cuda
```

A100 processes ~2000 steps/min — 50k steps ≈ 25 minutes.

### Training arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `S16` | Preset: `S16` (ViT-S/16, ~25M params) or `S32` (ViT-S/32) |
| `--reps_file` | `data/messidor2/train_reps.pt` | Precomputed representations (standard or packed) |
| `--save_dir` | `checkpoints/` | Checkpoint output directory |
| `--image_size` | `224` | Generator output resolution |
| `--batch_size` | `8` | Images per gradient step |
| `--grad_accum` | `1` | Gradient accumulation steps (effective batch = batch × accum) |
| `--lr` | `1e-4` | AdamW learning rate |
| `--warmup_steps` | `1000` | Linear LR warmup from 0 → lr |
| `--cfg_dropout` | `0.1` | Probability of replacing h with null_h during training |
| `--total_steps` | `100000` | Total gradient steps |
| `--save_interval` | `5000` | Save checkpoint every N steps |
| `--log_interval` | `100` | Log loss every N steps |
| `--resume_from` | — | Path to checkpoint to resume training |
| `--wandb_project` | — | Enable W&B logging to this project |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |

### Training duration guide

| Steps | Effective samples (batch=32) | What you typically see |
|---|---|---|
| 5 k | 160 k | Mean color and coarse brightness — orange blobs |
| 15 k | 480 k | Coarse structure — disc location, vessel quadrants |
| 30 k | 960 k | Vessel topology visible; CFG scale 1.5–2.0 usable |
| 50 k | 1.6 M | Fine vessel branches; CFG scale 2.0–3.0 usable |
| 100 k | 3.2 M | Micro-detail; full CFG range |

### Resuming

```bash
python scripts/train.py \
    --resume_from checkpoints/step_005000.pt \
    --device      cuda
```

Architecture is read from the checkpoint's `model_cfg` — no need to repeat model flags.

---

## Step 3 — Generate images

```bash
python scripts/sampling.py \
    --checkpoint  checkpoints/step_050000.pt \
    --cond_images data/messidor2/test/img1.png \
                  data/messidor2/test/img2.png \
    --out_dir     samples/ \
    --n_samples   4 \
    --cfg_scale   2.0 \
    --num_steps   50 \
    --device      cuda
```

Output: one PNG grid per conditioning image — leftmost tile is the conditioning input, remaining tiles are generated samples.

### Sampling arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Trained `.pt` checkpoint path |
| `--cond_images` | required | One or more conditioning image paths |
| `--out_dir` | `samples/` | Output directory |
| `--n_samples` | `4` | Samples to generate per conditioning image |
| `--cfg_scale` | `1.0` | Classifier-free guidance scale (see schedule below) |
| `--num_steps` | `50` | Heun ODE steps |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |

### CFG scale schedule

| Training steps completed | Recommended `--cfg_scale` |
|---|---|
| < 15 k | 1.0 (no guidance — null_h not yet trained) |
| 15–30 k | 1.5 |
| 30–50 k | 2.0–3.0 |
| 50 k+ | 3.0–5.0 |

`cfg_scale=1.0` is equivalent to no guidance (standard conditional forward pass). Values below 1.0 reduce conditioning strength; 0.0 ignores conditioning entirely.

---

## Programmatic usage

```python
import torch
from rcdm.encoder import load_encoder, build_transform
from rcdm.jit import JiT_S_16, FlowMatching
from PIL import Image

device = torch.device("cuda")

# --- Encoder ---
encoder   = load_encoder(device=device)
transform = build_transform(image_size=224)

# --- Model (from checkpoint) ---
ckpt  = torch.load("checkpoints/step_050000.pt", map_location=device)
cfg   = ckpt["model_cfg"]
model = JiT_S_16(image_size=cfg["image_size"])
model.load_state_dict(ckpt["ema"])     # use EMA weights for sampling
model.eval().to(device)

flow = FlowMatching()

# --- Encode a conditioning image ---
img = Image.open("data/messidor2/test/some_image.png").convert("RGB")
with torch.no_grad():
    out = encoder(pixel_values=transform(img).unsqueeze(0).to(device))
    h = out.last_hidden_state[:, 0, :]   # CLS token: (1, 384)
    h = h.expand(4, -1)                  # 4 samples from same conditioning

# --- Generate ---
noise = torch.randn(4, 3, 224, 224, device=device)
with torch.no_grad():
    samples = flow.sample(model, noise, h=h, num_steps=50, cfg_scale=2.0)
    # samples: (4, 3, 224, 224) in [-1, 1]

samples_uint8 = ((samples.clamp(-1, 1) + 1) * 127.5).byte()
```

---

## Troubleshooting

**`use_gated_mlp` mismatch — encoder gives non-deterministic representations**
Symptom: loss does not converge across runs; samples look different even with the same seed.
Fix: set `"use_gated_mlp": false` in `checkpoints/dinov3_vits16_tmp/config.json`, then delete and recompute `train_reps.pt`.

**`FileNotFoundError` for training images on Colab**
`train_reps.pt` stores absolute paths from your local machine. Use the packed format instead:
```bash
python scripts/pack_dataset.py --reps_file data/messidor2/train_reps.pt \
                                --out_file  data/messidor2/train_packed.pt
```
Then pass `--reps_file /content/train_packed.pt` to `train.py`.

**`RuntimeError: shape mismatch` when loading checkpoint**
The `model_cfg` in the checkpoint stores all architecture parameters. Read it to diagnose:
```python
import torch
print(torch.load("checkpoints/step_050000.pt", map_location="cpu")["model_cfg"])
```

**`AssertionError: Expected 384-dim reps`**
`train_reps.pt` was generated with a different encoder. Delete and rerun `precompute_reps.py`.

**Loss does not decrease past 0.28**
This is expected — MSE is dominated by easy components (color, brightness). The model is still learning fine vessel structure in the "flat" region of the loss curve. Do not stop training early; inspect sample quality at each checkpoint instead.

**Samples are near-black with `cfg_scale > 1`**
The null_h branch needs training before CFG extrapolation is useful. Use `--cfg_scale 1.0` until step 15k, then increase gradually.

---

## Citation

```bibtex
@article{bordes2022high,
  title   = {High Fidelity Visualization of What Your Self-Supervised Representation Knows About},
  author  = {Bordes, Florian and Balestriero, Randall and Vincent, Pascal},
  journal = {Transactions on Machine Learning Research},
  year    = {2022}
}

@inproceedings{li2024jit,
  title   = {Just-in-Time Diffusion: Generative Models with Deterministic Inference},
  author  = {Li, Tiankai and others},
  year    = {2024}
}
```
