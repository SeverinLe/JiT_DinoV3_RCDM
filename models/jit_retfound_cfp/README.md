# `jit_retfound_cfp` — not yet trained

Placeholder for the JiT-RCDM generator conditioned on **RETFound MAE ViT-L/16
(colour fundus weights)**, `h_dim = 1024`.

This run has not been executed.  Every checkpoint that existed before the
2026-08-03 restructure was verified by loading its `model_cfg`, and all of them
report `h_dim: 384` — i.e. DinoV3.  One file was *named*
`jit_rcdm_retfound_step0100000.pt`, which is where the impression of a RETFound
model came from; its contents say otherwise.  See `archive/MANIFEST.md`.

## Why this model matters to the study

The report's scientific spine is the comparison between SSL families:

| Encoder | SSL family | `h_dim` | Generator |
|---|---|---|---|
| DinoV3 ViT-S/16 | joint-embedding | 384 | `models/jit_dinov3/` ✅ |
| RETFound MAE ViT-L/16 | reconstructive (MAE), retinal domain | 1024 | this directory ❌ |
| ResNet-50 | supervised (control) | 2048 | not planned |

Without this model, E1–E4 report on one encoder only, and the report cannot
claim anything general about *SSL representation spaces* — only about one
DinoV3 checkpoint.  That limitation must be stated explicitly if the run does
not happen.

## How to produce it

```bash
# representations for every split (test and val are needed by E5)
for split in train val test; do
    python data/scripts/precompute_reps.py \
        --encoder  retfound_cfp \
        --data_dir data/raw/messidor2/$split \
        --out_file data/processed/messidor2/retfound_cfp/${split}_reps.pt \
        --batch_size 16 --device cuda
done

python scripts/train.py \
    --encoder     retfound_cfp \
    --reps_file   data/processed/messidor2/retfound_cfp/train_reps.pt \
    --save_dir    models/jit_retfound_cfp \
    --model       S16 \
    --batch_size  8 --lr 1e-4 --total_steps 100000 \
    --device      cuda --wandb_project jit-rcdm
```

`data/processed/messidor2/retfound_cfp/train_reps.pt` already exists (972 × 1024),
so only the val/test caches and the training run are outstanding.

Note the parameter-count consequence: `h_dim` 384 → 1024 enlarges only the
conditioning projector (`Linear(h_dim, 128)`), not the denoiser, so the compute
budget is essentially the same as the DinoV3 run.  Encoding is slower — ViT-L/16
against ViT-S/16 — but that cost is paid once, during precompute.

After training, record the run in `models/README.md` and write a
`training_log.md` here with hardware and wall-clock for report §4.5.
