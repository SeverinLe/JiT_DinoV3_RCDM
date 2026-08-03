# `jit_dinov3` — training log

Source data for §4.5 (hardware and compute) of the report.  **Fields marked
`TODO` must be filled from the W&B run before it is written up** — the rubric's
top band requires precise hardware and compute time, not estimates.

## Runs

| Field | `final.pt` | `step0055000.pt` |
|---|---|---|
| W&B run id | `c48aue7x` | `c48aue7x` (same run) |
| Steps | 100 000 | 55 000 |
| Encoder | DinoV3 ViT-S/16, frozen, `h_dim` 384 | same |
| Preset | `JiT_S_16` — hidden 384, depth 12, heads 6, patch 16, `cond_dim` 128 | same |
| Parameters | 25.7 M | 25.7 M |
| Image size | 224 × 224 | 224 × 224 |
| Objective | flow matching, x-prediction, `t ~ sigmoid(N(-0.8, 0.8))` | same |
| EMA | decay 0.9999, stored in checkpoint | same |
| Optimiser | AdamW, β₂ = 0.95, linear warmup | same |
| Batch size | TODO | TODO |
| Learning rate | TODO | TODO |
| CFG dropout | TODO (`--cfg_dropout`, default 0.1) | TODO |
| Dataset | Messidor-2 train, 972 images | same |
| Hardware | Google Colab — TODO: GPU model, VRAM, CPU | same |
| Wall-clock | TODO | TODO |
| GPU-hours | TODO | TODO |
| Checkpoint sha256 | see `final.cfg.json` | see `step0055000.cfg.json` |

The run was executed on Colab, so there is no local `wandb/run-*-c48aue7x`
directory — pull the configuration and system metrics from the W&B web UI.

## Provenance note

Both files were previously named `jit_rcdm_retfound_step0100000.pt` and
`jit_rcdm_step0055000.pt`.  The first name was wrong: the stored `model_cfg`
reads `h_dim: 384`, which is DinoV3 (RETFound would be 1024).  Renamed during
the 2026-08-03 restructure; `scripts/train.py` now writes `encoder` into
`model_cfg` so the ambiguity cannot recur.

## Related runs (checkpoints deleted, metrics still in W&B)

| Run | What it was | Why it is not here |
|---|---|---|
| `oavlyyc5` | 50 000 steps with an incorrect `h` | Known-bad; kept as a documented negative result, not as weights |
| `nmf6l3rz` | hidden 768 / `cond_dim` 768 | Capacity ablation |
| `rctoipbq` | `cond_dim` 64 | Bottleneck-width ablation |
| `6gecr1sw` | "S32" (stored config says patch 16) | Ablation; name and config disagree |

See `archive/MANIFEST.md` for sizes and full configurations.
