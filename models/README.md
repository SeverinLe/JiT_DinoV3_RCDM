# Trained generators

JiT-RCDM generators trained in this project — the *outputs* of the study.  Not to
be confused with `encoders/`, which holds the frozen pretrained SSL encoders that
are the *objects* of the study.

One generator is trained per encoder, because the conditioning projector and every
adaLN-Zero MLP are sized for a specific representation width.  A generator cannot
be paired with an encoder it was not trained on; `experiments/common.py` enforces
this by comparing `h_dim` and raises rather than producing plausible nonsense.

Weights are **not** tracked in git (400 MB – 1.6 GB each).  Each `*.pt` has a
committed `*.cfg.json` sidecar recording its architecture, training step, W&B run
id and sha256, so a checkpoint can always be identified and verified.

## Available models

| Directory | Encoder | `h_dim` | Preset | Steps | EMA | W&B run | Status |
|---|---|---|---|---|---|---|---|
| `jit_dinov3/` | DinoV3 ViT-S/16 | 384 | `JiT_S_16` (hidden 384, depth 12, heads 6, patch 16, cond_dim 128) | 100 000 | yes | `c48aue7x` | **primary** |
| `jit_dinov3/step0055000.pt` | DinoV3 ViT-S/16 | 384 | same | 55 000 | yes | `c48aue7x` | training-length ablation |
| `jit_retfound_cfp/` | RETFound MAE ViT-L/16 (CFP) | 1024 | — | — | — | — | **not yet trained** |

### `jit_dinov3/final.pt` — the main model

~25.7 M parameters.  Trained on 972 Messidor-2 fundus images at 224 px with
flow matching (x-prediction), sampled with a 50-step Heun ODE and classifier-free
guidance via a learnable `null_h`.

A naming caveat worth knowing: this file was previously called
`jit_rcdm_retfound_step0100000.pt`, but its stored `model_cfg` reads
`h_dim: 384` — it is a **DinoV3** model.  The filename was wrong.  Since then
`scripts/train.py` writes `encoder` into `model_cfg` so this cannot recur; the
two checkpoints here predate that change and their sidecars record the encoder
as inferred from `h_dim`.  See `archive/MANIFEST.md`.

Load it with:

```python
from experiments.common import load_generator, resolve_device
model, flow, cfg = load_generator("models/jit_dinov3/final.pt", resolve_device("cuda"))
```

EMA weights are applied automatically — every reported number should come from
the same weights, and the JiT paper (Tab. 9) reports EMA at decay 0.9999 as the
best-FID configuration.

## Obtaining the weights

They are not in the repository.  Either

- download the checkpoint artefacts from the W&B run listed above, or
- request them from the author (sha256 values in the `*.cfg.json` sidecars let
  you verify what you receive).

## Training a new generator

```bash
# 1. cache the representations (once per encoder and split)
python data/scripts/precompute_reps.py \
    --encoder  retfound_cfp \
    --data_dir data/raw/messidor2/train \
    --out_file data/processed/messidor2/retfound_cfp/train_reps.pt

# 2. train
python scripts/train.py \
    --encoder     retfound_cfp \
    --reps_file   data/processed/messidor2/retfound_cfp/train_reps.pt \
    --save_dir    models/jit_retfound_cfp \
    --model       S16 \
    --batch_size  8 --lr 1e-4 --total_steps 100000 --device cuda
```

`train.py` refuses to start if `--encoder` disagrees with the representation
cache, so the mismatch that produced the misnamed checkpoint above cannot happen
again.
