# Frozen encoders

The pretrained SSL encoders under study — the *inputs* of the project, as
opposed to `models/`, which holds the generators trained on top of them.

All are frozen (`requires_grad=False`, `eval()`), consume 224 × 224
ImageNet-normalised input, and expose the same interface through
`rcdm/encoders/`:

```python
from rcdm.encoders import get_encoder, build_transform
encoder = get_encoder("dinov3", device="cuda")
h = encoder(build_transform()(img).unsqueeze(0).to("cuda"))   # (1, encoder.h_dim)
```

| Registry name | Model | SSL family | `h_dim` | Weights | Tracked |
|---|---|---|---|---|---|
| `dinov3` | DinoV3 ViT-S/16 | joint-embedding | 384 | `dinov3_vits16/` | no (86 MB) |
| `retfound_cfp` | RETFound MAE ViT-L/16, colour-fundus weights | reconstructive (MAE) | 1024 | `retfound_cfp/` | no (3.7 GB) |
| `resnet50` | ImageNet-supervised ResNet-50 | supervised (control) | 2048 | torchvision zoo | n/a |

The three-way split — joint-embedding vs. reconstructive vs. supervised — is the
scientific spine of the study: it is what lets conclusions be about *SSL
representation spaces* rather than about one checkpoint.

## DinoV3 ViT-S/16

Expected layout:

```
encoders/dinov3_vits16/
├── config.json
└── model.safetensors
```

Loaded with `local_files_only=True`, so no network request is ever made and the
weights are pinned for reproducibility.  `download_dinov3_vits16.sh` fetches
them.

### The `use_gated_mlp` trap — read before running anything

The checkpoint was trained with a standard 2-projection FFN, but the shipped
`config.json` sets `"use_gated_mlp": true`, which makes transformers expect a
3-projection gated FFN.  The missing `gate_proj` is then **randomly initialised
on every load**, so `h` differs between runs and every downstream number becomes
irreproducible — silently.

Fix it once, in `config.json`:

```json
"use_gated_mlp": false
```

`rcdm/encoders/dinov3.py` refuses to load when the flag is `true` rather than
letting the corruption through.

## RETFound MAE ViT-Large/16

Expected layout:

```
encoders/retfound_cfp/
├── RETFound_mae_natureCFP.pth
├── config.json
└── README.md
```

Access-gated: request via <https://github.com/rmaphoh/RETFound_MAE> or the
HuggingFace model card, accept the CC BY-NC 4.0 licence, then download.
**Non-commercial use only.**

Two weight sets exist and are *not* interchangeable:

| File | Pre-training corpus | Use with |
|---|---|---|
| `RETFound_mae_natureCFP.pth` | colour fundus photography | Messidor-2 ✅ (what this project uses) |
| `RETFound_mae_natureOCT.pth` | optical coherence tomography | OCT datasets |

Both are ViT-L/16 with a 1024-dim CLS token; only the corpus differs.  Any
report must state which one produced each result.

The `.pth` is a raw MAE pre-training checkpoint containing both the encoder and
the MAE decoder.  `rcdm/encoders/retfound.py` drops the `decoder_*` and
`mask_token` keys before loading and then **raises** if any encoder key is still
missing or unexpected — a partial load would mean silently random weights.

## ResNet-50

No local weights: `torchvision.models.resnet50(weights=DEFAULT)` downloads to
`~/.cache/torch` on first use.  The classification head is replaced with
`Identity`, giving the 2048-dim avgpool trunk output — the same representation
RCDM used originally.

## Adding an encoder

Create `rcdm/encoders/<name>.py` exposing `H_DIM` and `load(device, **kwargs)`
returning a frozen module with `.name`, `.h_dim` and
`forward(x) -> (B, h_dim)`, then add one line to `ENCODERS` in
`rcdm/encoders/__init__.py`.  Nothing else changes: the dataset, denoiser,
trainer and every experiment are parametrised by `h_dim` throughout.
