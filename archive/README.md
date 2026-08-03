# Archive

Superseded code kept for provenance.  Nothing here is imported by the live
pipeline; it exists so that every script that ever produced a result in this
project remains readable.

```
archive/
├── MANIFEST.md                 what was removed in the restructure, and where to recover it
├── README_pre_restructure.md   the previous root README (architecture walkthrough)
├── scripts/
│   ├── main_branch/            scripts/ + rcdm/ as they stood on main @ 6733acc
│   └── worktree_branch/        scripts/ + rcdm/ from claude/silly-faraday-d8512b
└── notebooks/                  the Colab training notebooks
```

## Why `worktree_branch/` matters

That branch was **local-only** — never pushed to `origin`.  It held the only
copies of the invariance and difference-matrix probes, which became
`experiments/E3_invariance.py` and informed `E2_sample_variability.py`.  Copying
them here is what made removing the worktree safe.

Files unique to it: `invariance_images.py`, `invariance_probe.py`,
`invariance_probe_noise.py`, `diff_matrix.py`, `encoder_retfound.py`,
`precompute_reps_retfound.py`, `sampling_retfound.py`, and the working JiT
trainer.

## What replaced what

| Archived | Live equivalent |
|---|---|
| `encoder.py` (two divergent copies) + `encoder_retfound.py` | `rcdm/encoders/` registry |
| `precompute_reps.py` + `precompute_reps_retfound.py` | `data/scripts/precompute_reps.py --encoder` |
| `sampling.py` + `sampling_retfound.py` | `experiments/sample_grid.py --encoder` |
| `fid_eval.py` | `experiments/E1_fidelity.py` (adds the rank/MRR metric) |
| `diff_matrix.py` | `experiments/E2_sample_variability.py` |
| `invariance_images.py` + `invariance_probe.py` | `experiments/E3_invariance.py` |
| `train.py` (worktree copy) | `scripts/train.py` |

`main_branch/train.py` is **not** the trainer that produced the results: it was a
half-migrated file importing `guided_diffusion` alongside the JiT modules and did
not run.  The worktree copy is the real one.

`diff_matrix.py` is worth keeping in view — its pairwise difference-matrix
visualisation is richer than what `E2_sample_variability.py` currently produces
and could be folded back in if the report needs it.
