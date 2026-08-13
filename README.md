# JiT-RCDM — Probing SSL representations of retinal images through generation

Master's practical work. **Research question: what does a frozen self-supervised
encoder actually keep about a retinal scan, and what does it throw away?**

Linear probes answer a narrower question — whether information is *decodable* —
not whether it is *present*.  Following Bordes et al. (2022), this project
inverts the encoder instead: a diffusion generator is conditioned on a frozen
representation `h`, and sampling several images from one `h` makes the encoded
content visible (what stays constant) and the discarded content visible (what
varies).  The motivation is downstream: understanding what a frozen retinal
encoder carries tells you where a grading classifier built on it will hit its
ceiling, and which nuisance factors it can silently latch onto.

The implementation merges two upstream frameworks:

- **RCDM** ([facebookresearch/RCDM](https://github.com/facebookresearch/RCDM)) — conditioning a diffusion model on an SSL representation `h` instead of a class label
- **JiT** ([LTH14/JiT](https://github.com/LTH14/JiT)) — a plain ViT denoiser trained with flow matching

Applied to **Messidor-2** diabetic-retinopathy fundus images (972 train / 246 val
/ 527 test, 5 grades) with frozen encoders from two different SSL families.

`CHANGES.md` documents every deviation from the two upstream repositories.
`REPORT_OUTLINE.md` maps the experiments to the report sections.

---

## Repository layout

```
rcdm/                   library code
  jit.py                ViT denoiser + FlowMatching (training loss, Heun sampler)
  conditioning.py       ConditioningProjector, AdaLNZero, RMSNorm
  dataset.py            (image, h) pairs from a representation cache
  encoders/             the frozen encoders under study — registry + one module each

data/
  scripts/              precompute_reps.py, pack_dataset.py
  raw/messidor2/        images, ImageFolder layout (not tracked)
  processed/messidor2/  <encoder>/{train,val,test}_reps.pt (not tracked)

encoders/               frozen pretrained SSL weights — study INPUTS  (not tracked)
models/                 trained JiT-RCDM generators — study OUTPUTS   (not tracked)

scripts/train.py        train a generator on cached representations
notebooks/              Colab notebooks (training, t-sampler ablation)
experiments/            E1-E5 probes + sample_grid, results/ per experiment
report/                 figures and tables selected for the write-up
wandb/                  run logs (compute data for the report)
archive/                superseded scripts + MANIFEST.md of what was removed
```

The `encoders/` vs `models/` split is deliberate: the frozen encoder is the
independent variable of the study, the generator trained on it is the dependent
one.  One generator is trained per encoder, since the conditioning path is sized
for a specific `h_dim`.

---

## Setup

```bash
git clone <this repo> && cd master_implementation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13 on macOS/MPS for inference and Colab/A100 for training.
If you need a specific CUDA build, install PyTorch first from
<https://pytorch.org/get-started/locally/>.

### Data

Messidor-2 is access-controlled and is **not** redistributed here.  Request it
from <https://www.adcis.net/en/third-party/messidor2/> and arrange it as

```
data/raw/messidor2/{train,val,test}/<grade>/*.png
```

with grades `anodr`, `bmilddr`, `cmoderatedr`, `dseveredr`, `eproliferativedr`.
Details and exact split sizes are in `data/README.md`.

### Encoder weights

Both are gated downloads; see `encoders/README.md` for the procedure, the
expected directory layout, and the **`use_gated_mlp: false` fix** for DinoV3 —
without it, transformers randomly initialises a missing projection on every load
and `h` differs between runs.

### Trained generators

Not in the repository (400 MB each).  `models/README.md` lists each model with
its W&B run id and sha256; every `*.pt` has a committed `*.cfg.json` sidecar.

---

## Pipeline

```
raw images ──► data/scripts/precompute_reps.py ──► <split>_reps.pt
                                                        │
                                                        ▼
                                              scripts/train.py ──► models/<name>/final.pt
                                                        │
                                                        ▼
                                              experiments/E1-E5 ──► results/<experiment>/…
```

```bash
# 1. cache representations (once per encoder and split)
for split in train val test; do
    python data/scripts/precompute_reps.py \
        --encoder  dinov3 \
        --data_dir data/raw/messidor2/$split \
        --out_file data/processed/messidor2/dinov3/${split}_reps.pt
done

# 2. train the generator
python scripts/train.py \
    --encoder dinov3 \
    --reps_file data/processed/messidor2/dinov3/train_reps.pt \
    --save_dir  models/jit_dinov3 \
    --model S16 --batch_size 8 --lr 1e-4 --total_steps 100000 --device cuda

# 3. probe (see the next section)
```

The encoder never runs inside the training loop — representations are cached
once, so training, sampling and every probe see byte-identical `h` for the same
image.  `train.py` refuses to start if `--encoder` disagrees with the cache.

---

## Reproducing the report

Every experiment writes a timestamped directory under
`experiments/results/<experiment>/<encoder>/<tag>/` containing `run_config.json`
(all arguments, RNG seed, git commit, checkpoint sha256, library versions),
`metrics.csv`, `summary.json` and `figures/`.  Any figure in the report can
therefore be traced back to the exact state that produced it.

| Report item | Experiment | Command |
|---|---|---|
| Fig. 2 — qualitative grids | `sample_grid` | `python experiments/sample_grid.py --checkpoint models/jit_dinov3/final.pt --encoder dinov3 --cond_images data/raw/messidor2/test/*/*.png --n_samples 4 --cfg_scale 3.0 --seed 0` |
| Table 1, Fig. 5 — RQ1 fidelity | `E1_fidelity` | `python experiments/E1_fidelity.py --checkpoint models/jit_dinov3/final.pt --encoder dinov3 --n_per_class 40 --seed 0` |
| RQ2 — encoded vs. discarded | `E2_sample_variability` | `python experiments/E2_sample_variability.py --checkpoint models/jit_dinov3/final.pt --encoder dinov3 --n_per_class 6 --n_samples 8 --seed 0` |
| Fig. 3 — RQ3 invariance | `E3_invariance` | `python experiments/E3_invariance.py --checkpoint models/jit_dinov3/final.pt --encoder dinov3 --n_per_class 30 --n_visual 1 --seed 0` |
| Fig. 4 — RQ4 dimension structure | `E4_dimension_structure` | `python experiments/E4_dimension_structure.py --checkpoint models/jit_dinov3/final.pt --encoder dinov3 --n_queries 10 --n_dims 64 --seed 0` |
| Table 2 — RQ5 downstream probe | `E5_downstream_probe` | `python experiments/E5_downstream_probe.py --encoder dinov3 --dimensions_file experiments/results/E4_dimension_structure/dinov3/<tag>/dimensions.json --n_seeds 5` |

Add `--device cuda` (or `mps`) throughout.  Metrics that need error bars across
runs (E1's FID, E2's variability) should be repeated over several `--seed` values
and aggregated; E1's MRR, E3's cosine similarities and E5's probe scores compute
their own confidence intervals internally.

`experiments/README.md` explains what each probe measures and how to read it.

---

## Architecture

```
conditioning image (fundus photo, any resolution)
   │  Resize(224) + CenterCrop(224) + ImageNet normalise
   ▼
frozen SSL encoder  ──────────────►  h  (B, h_dim)
   (DinoV3 ViT-S/16 → 384 | RETFound ViT-L/16 → 1024 | ResNet-50 → 2048)
   │
   ▼  ConditioningProjector: Linear(h_dim → 128) + SiLU
h_proj (B, 128)  +  t_emb (B, 128)  =  c
   │
   ▼  broadcast to every block
JiT ViT denoiser (JiT_S_16: hidden 384, depth 12, heads 6, patch 16, ~25.7 M params)
   PatchEmbed → 12 × JiTBlock[adaLN-Zero(c) · Attention(RoPE, qk-norm) · SwiGLU] → FinalLayer
   ▼
x_pred (B, 3, 224, 224)
```

**Conditioning.** `c = h_proj + t_emb` is computed once and fed to every block's
adaLN-Zero MLP, which emits shift/scale/gate for both the attention and the FFN
branch — 25 modulations per forward pass (12 blocks × 2 + final layer).  The
adaLN output projection is zero-initialised, so every block starts as the
identity and conditioning engages gradually as the gates depart from zero.  RCDM
used conditional BatchNorm, which modulates `(B, C, H, W)` feature maps and has
no meaning for a ViT's `(B, N, D)` token sequences.

**Encoder input is always 224 px**, independent of the generator's resolution:
a /16 ViT has a fixed 14×14 positional grid (224/16), and any other input size
silently interpolates or corrupts those embeddings.

**Training.** Flow matching with x-prediction:
`z_t = t·x + (1−t)·ε`, `t ~ sigmoid(N(−0.8, 0.8))`, `L = MSE(model(z_t, t, h), x)`.
AdamW with β₂ = 0.95, linear warmup, EMA at decay 0.9999.

**Sampling.** 50-step Heun ODE with classifier-free guidance against a learnable
`null_h` parameter (trained via `--cfg_dropout`, included in the EMA), blending
at the `x_pred` level.

`cond_dim = 128` is our own bottleneck, inherited from neither upstream repo
(RCDM used 512, JiT used `hidden_dim`); it regularises the conditioning path for
a <1k-image dataset and shrinks every block's adaLN MLP.  Full rationale and the
complete list of deviations from RCDM and JiT are in `CHANGES.md`.

---

## Status

| Component | State |
|---|---|
| DinoV3-conditioned generator | trained, 100k steps — `models/jit_dinov3/final.pt` |
| RETFound-conditioned generator | **not yet trained** — see `models/jit_retfound_cfp/README.md` |
| E1–E5 probes | implemented; E5 additionally needs val/test representation caches |
| OCT modality | not started — the encoder wiring exists, the data does not |

## Citation

```
Bordes, Balestriero, Vincent (2022). High Fidelity Visualization of What Your
  Self-Supervised Representation Knows About. arXiv:2112.09164
Li, Katabi, He (2024). Return of Unconditional Generation: A Self-supervised
  Representation Generation Method. arXiv:2312.03701
Zhou et al. (2023). A foundation model for generalizable disease detection from
  retinal images. Nature 622, 156-163.
Decencière et al. (2014). Feedback on a publicly distributed image database:
  the Messidor database. Image Analysis & Stereology 33(3).
```
