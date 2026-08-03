# Experiments

One script per research question.  Each writes a self-describing run directory:

```
experiments/results/<experiment>/<encoder>/<tag>/
├── run_config.json     every CLI argument, RNG seed, git commit,
│                       checkpoint sha256, library versions, timestamp
├── metrics.csv         per-image or per-condition raw measurements
├── summary.json        aggregates, confidence intervals, test statistics
└── figures/            publication-ready PNGs at 300 dpi
```

`run_config.json` is written *before* any compute starts, so even an interrupted
run leaves a record of what was attempted.  Nothing under `results/` is tracked
in git; figures selected for the write-up are copied into `report/figures/`.

## The probes

| Script | RQ | Question | Primary output |
|---|---|---|---|
| `E1_fidelity.py` | RQ1 | Is `h` invertible — can the generator reconstruct the scan it came from? | MRR + mean rank with bootstrap CI, FID |
| `E2_sample_variability.py` | RQ2 | What does `h` encode vs. discard? | per-pixel std, pairwise SSIM, variability maps |
| `E3_invariance.py` | RQ3 | Which transformations is `h` invariant to? | cosine similarity per transform ± CI, Wilcoxon + Holm |
| `E4_dimension_structure.py` | RQ4 | Do individual dimensions carry separable factors? | masking/substitution grids, `dimensions.json` |
| `E5_downstream_probe.py` | RQ5 | Do the generative findings predict classifier behaviour? | balanced accuracy ± 95% CI, McNemar |
| `sample_grid.py` | — | Qualitative figures | `[real \| k samples]` grids |

`common.py` holds the shared plumbing: run-directory creation, provenance
capture, seeding, generator/encoder loading, and the matplotlib style.

## How the probes fit together

E1 establishes that the inversion is faithful enough to be read at all — without
it, nothing downstream is interpretable.  E2 and E3 then read the representation
in two directions: what varies across samples from one `h` (discarded), and what
moves `h` itself (non-invariance).  E4 asks whether the structure is coordinate-
aligned enough to edit.  E5 is the external check: it takes E4's dimension
indices and E3's non-invariant transforms and asks whether they predict what a
linear probe can and cannot do.

That last link matters more than it looks.  E1–E4 measure what a *generator* can
render from `h`; a linear probe measures what a *downstream model* can decode.
These come apart in both directions, and RQ5 is the test of whether the
generative probe earns its keep as a diagnostic.

## Reading a result honestly

**The inversion confound.** A feature missing from the samples can mean the
information is absent from `h`, *or* that the generator cannot render it.  With
972 training images the second explanation is always live.  E1's fidelity metric
bounds it (a faithful inversion makes "the generator cannot render it" less
plausible) and E5 bounds it from the other side, but the bound is partial and the
report says so.

**FID on retinal images.** The Inception backbone is ImageNet-trained and poorly
calibrated for this domain, and the estimator is biased at a few hundred images.
Treat FID as a relative signal between configurations, never as an absolute
quality claim.

**Error bars.** E1, E3 and E5 compute their own confidence intervals
(bootstrap over images for E1/E3, over seeds for E5).  E2's variability metrics
and E1's FID need repeated `--seed` runs to get a spread — a single FID number
carries no error bar.

## Conventions

- `--seed` on every script; recorded in `run_config.json`.
- `--tag` names a run directory; omitted, it defaults to a UTC timestamp.
- `--device cpu|cuda|mps`, falling back to CPU with a warning if unavailable.
- `--encoder` is checked against the checkpoint's `h_dim`; a mismatch raises
  rather than producing plausible nonsense.
- EMA weights are applied automatically when present, so every number comes from
  the same weights.

## Dependency between runs

E5 consumes `dimensions.json` from an E4 run and, for condition C, a
representation cache built from transformed images:

```bash
python experiments/E4_dimension_structure.py --checkpoint models/jit_dinov3/final.pt \
    --encoder dinov3 --n_queries 10 --n_dims 64 --seed 0

python experiments/E5_downstream_probe.py --encoder dinov3 \
    --dimensions_file experiments/results/E4_dimension_structure/dinov3/<tag>/dimensions.json \
    --n_seeds 5
```

E5 also needs `val`/`test` representation caches, which do not exist yet — see
`data/README.md` for the loop that builds them.
