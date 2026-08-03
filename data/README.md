# Data

```
data/
├── scripts/                     preparation stage
│   ├── precompute_reps.py       frozen encoder -> cached representations
│   └── pack_dataset.py          embed images into the cache for Colab
├── raw/messidor2/               source images (not tracked)
│   ├── train/<grade>/*.png
│   ├── val/<grade>/*.png
│   └── test/<grade>/*.png
└── processed/messidor2/         representation caches (not tracked)
    ├── dinov3/{train,val,test}_reps.pt
    └── retfound_cfp/{train,val,test}_reps.pt
```

Neither `raw/` nor `processed/` is tracked in git: Messidor-2 is
access-controlled and may not be redistributed, and the caches are derived
artefacts (1.5 MB – 150 MB each) that any user can rebuild in minutes.

## Messidor-2

Diabetic-retinopathy screening images, colour fundus photography.
Obtain from <https://www.adcis.net/en/third-party/messidor2/> (registration
required) and cite Decencière et al. (2014) and Abràmoff et al. (2013).

The version used here is the preprocessed release, arranged into five
severity-graded folders in ImageFolder layout.

| Split | Total | `anodr` | `bmilddr` | `cmoderatedr` | `dseveredr` | `eproliferativedr` |
|---|---|---|---|---|---|---|
| train | 972 | 568 | 151 | 193 | 41 | 19 |
| val | 246 | 143 | 38 | 49 | 11 | 5 |
| test | 527 | 306 | 81 | 105 | 23 | 11 |

Folder names sort into severity order (`a` … `e`) — no-DR, mild, moderate,
severe, proliferative.

**Two properties that must be reported alongside any result on this data.**

*Class imbalance.* 58% of the training split is grade `anodr`, and the two most
advanced grades have 19 and 41 images.  Predicting the majority class alone
scores 0.58 plain accuracy, which is why every downstream metric in this project
is balanced accuracy or macro-AUC.  For the generator it means advanced
pathology is represented by a few dozen examples, and any claim about how well
lesions are rendered has to be read against that.

*Split protocol.* The splits are inherited from the preprocessed release and are
image-level.  Messidor-2 contains both eyes of the same patient, so
patient-level leakage between splits cannot be excluded.  This bounds how far
the E5 probe numbers can be pushed and belongs in the report's limitations.

## Preprocessing

No offline preprocessing is applied — resizing and normalisation happen in the
data pipeline, and deliberately differ between the two consumers:

| Consumer | Pipeline | Range |
|---|---|---|
| encoder (`rcdm/encoders/transforms.py`) | Resize(224) → CenterCrop(224) → ToTensor → Normalize(ImageNet) | ImageNet mean/std |
| generator (`rcdm/dataset.py`) | Resize/Crop to `image_size` → ToTensor → scale | `[-1, 1]` |

They are never mixed: `h` is computed in encoder space, while the diffusion
target `x` lives in `[-1, 1]`.

## Building the representation caches

```bash
for split in train val test; do
    python data/scripts/precompute_reps.py \
        --encoder  dinov3 \
        --data_dir data/raw/messidor2/$split \
        --out_file data/processed/messidor2/dinov3/${split}_reps.pt \
        --batch_size 32 --device cuda
done
```

Cache format:

```python
{
  "paths":   list[str],    # N image paths, sorted
  "labels":  list[str],    # parent folder = class, aligned with paths
  "reps":    Tensor(N, D), # float32
  "encoder": str,          # registry name, e.g. "dinov3"
  "h_dim":   int,          # D
}
```

`reps[i]` belongs to `paths[i]`; the whole pipeline depends on that alignment,
and `precompute_reps.py` asserts it before saving.  Consumers check the
`encoder` and `h_dim` fields and refuse to run on a mismatch.

Only the `train` split is needed to train a generator.  **E5 additionally needs
`val` and `test`** — currently only `train_reps.pt` exists for both encoders.

`pack_dataset.py` embeds the images themselves into the cache as uint8, which
removes the dependency on absolute image paths when training on Colab.
