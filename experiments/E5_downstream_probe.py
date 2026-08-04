"""
experiments/E5_downstream_probe.py — RQ5: do the generative findings predict
downstream classifier behaviour?

E1-E4 measure what the *generator* can render from h.  A linear probe measures
something different: what a *simple downstream model* can decode from h.  The two
come apart in both directions, and RQ5 is the test of whether the generative
probe earns its keep as a diagnostic.

Conditions compared (all on the same frozen representations, no fine-tuning):

  A  full            the complete representation — the baseline ceiling
  B  masked          the E4-selected dimensions zeroed.  If E4's generations
                     showed those dimensions carrying the lesion signal,
                     accuracy should fall; if they carried background, it
                     should not.  The prediction is made from the pictures and
                     tested by the number.
  C  transformed     representations of test images under a transform E3 found
                     the encoder is *not* invariant to.  A large drop means the
                     encoder carries acquisition/device signal that a deployed
                     classifier would silently depend on.

Metrics and statistics
----------------------
Balanced accuracy and macro-AUC, never plain accuracy: Messidor-2's training
split is 58% no-DR, so predicting the majority class alone scores 0.58.
Reported as mean ± 95% CI over --n_seeds probe fits, with a paired McNemar test
between each condition and the baseline on the shared test set.

Requirements
------------
Needs representations for all three splits, which the default cache does not
include.  Build them first:

    for split in train val test; do
        python data/scripts/precompute_reps.py --encoder dinov3 \
            --data_dir data/raw/messidor2/$split \
            --out_file data/processed/messidor2/dinov3/${split}_reps.pt
    done

Usage:
    python experiments/E5_downstream_probe.py \
        --encoder   dinov3 \
        --reps_dir  data/processed/messidor2/dinov3 \
        --dimensions_file experiments/results/E4_dimension_structure/dinov3/<tag>/dimensions.json \
        --n_seeds   5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import (
    REPO_ROOT,
    create_run_dir,
    resolve_device,
    save_metrics,
    set_plot_style,
    set_seed,
)

EXPERIMENT = "E5_downstream_probe"


def load_split(reps_dir: Path, split: str, encoder: str):
    """
    Load one split's representation cache.

    Returns:
        (reps, labels) — (N, D) float tensor and (N,) array of class-name strings.
    """
    path = Path(reps_dir) / f"{split}_reps.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. E5 needs train/val/test representations — see the "
            "module docstring for the precompute_reps.py loop that builds them."
        )
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("encoder") and cache["encoder"] != encoder:
        raise ValueError(f"{path} was built with encoder '{cache['encoder']}', not '{encoder}'")
    labels = cache.get("labels")
    if labels is None:
        # Older caches predate the "labels" key; the class is the parent folder.
        labels = [Path(p).parent.name for p in cache["paths"]]
    return cache["reps"].float().numpy(), np.asarray(labels)


def fit_probe(x_train, y_train, x_test, seed: int, max_iter: int = 2000):
    """
    Fit a multinomial logistic regression probe on frozen representations.

    Standardisation matters: representation dimensions have very different
    scales, and an unstandardised L2 penalty would effectively regularise them
    unequally.

    Returns:
        (predictions, probabilities) on x_test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            # Class weighting rather than resampling: the tail grades have as few
            # as 19 training images and resampling would duplicate them heavily.
            class_weight="balanced",
            random_state=seed,
        ),
    )
    probe.fit(x_train, y_train)
    return probe.predict(x_test), probe.predict_proba(x_test)


def score(y_true, y_pred, y_proba, classes) -> dict:
    """Balanced accuracy + macro one-vs-rest AUC."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    metrics = {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}
    try:
        metrics["macro_auc"] = float(roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=classes
        ))
    except ValueError:
        # Happens when a class is absent from the test split.
        metrics["macro_auc"] = float("nan")
    return metrics


def mcnemar(baseline_correct: np.ndarray, condition_correct: np.ndarray) -> dict:
    """
    Paired McNemar test between two classifiers on the same test set.

    Uses the exact binomial test — the discordant counts here are small enough
    that the chi-square approximation is not safe.
    """
    only_baseline = int(np.sum(baseline_correct & ~condition_correct))
    only_condition = int(np.sum(~baseline_correct & condition_correct))
    result = {"n_only_baseline": only_baseline, "n_only_condition": only_condition}
    try:
        from scipy.stats import binomtest

        n = only_baseline + only_condition
        result["p_value"] = float(binomtest(only_baseline, n, 0.5).pvalue) if n else 1.0
    except ImportError:
        result["p_value"] = None
    return result


def ci95(values: np.ndarray) -> tuple:
    """Mean and normal-approximation 95% CI over repeated fits."""
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    if len(values) < 2:
        return mean, mean, mean
    sem = float(np.nanstd(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def bootstrap_test_ci(y_true, y_pred, y_proba, classes,
                      n_boot: int = 10_000, seed: int = 0) -> dict:
    """
    Percentile bootstrap CIs for balanced accuracy and macro-AUC, resampling the
    *test set*.

    These are the intervals that belong on the figure.  Refitting the probe under
    different RNG seeds produces bit-identical results — sklearn's lbfgs solver is
    deterministic given the data — so a CI computed over seeds is zero-width and
    misrepresents the uncertainty as nil.  The real variability is in which images
    happen to be in the test set, and that is what resampling here estimates.

    Stratified resampling keeps each class's count fixed, so both metrics stay
    defined even though the two most advanced grades have only 23 and 11 test
    images.  That scarcity is exactly why the intervals come out wide, and the
    report should say so rather than quoting the point estimate alone.

    Returns:
        {"balanced_accuracy": {...}, "macro_auc": {...}} with mean and ci95.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    class_indices = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]

    accuracies, aucs = [], []
    for _ in range(n_boot):
        picked = np.concatenate([
            rng.choice(idx, size=len(idx), replace=True) for idx in class_indices
        ])
        accuracies.append(balanced_accuracy_score(y_true[picked], y_pred[picked]))
        try:
            aucs.append(roc_auc_score(y_true[picked], y_proba[picked],
                                      multi_class="ovr", average="macro",
                                      labels=classes))
        except ValueError:
            aucs.append(np.nan)

    def _summarise(point, draws):
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        return {"mean": float(point), "ci95": [float(lo), float(hi)],
                "ci_source": "stratified bootstrap over test set"}

    try:
        auc_point = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                  average="macro", labels=classes)
    except ValueError:
        auc_point = float("nan")

    return {
        "balanced_accuracy": _summarise(
            balanced_accuracy_score(y_true, y_pred), accuracies),
        "macro_auc": _summarise(auc_point, aucs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E5 — downstream linear probe")
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--reps_dir", default="data/processed/messidor2/dinov3",
                        help="Directory holding {train,val,test}_reps.pt")
    parser.add_argument("--dimensions_file", default=None,
                        help="dimensions.json from an E4 run — enables condition B")
    parser.add_argument("--transformed_reps", default=None,
                        help="test_reps.pt built from transformed images — "
                             "enables condition C. Produce it by running "
                             "precompute_reps.py over a transformed copy of the "
                             "test split (see E3 for the transform set).")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Probe refits. Note the solver is deterministic, so "
                             "this checks fit stability rather than producing the "
                             "error bars — those come from --n_boot.")
    parser.add_argument("--n_boot", type=int, default=10_000,
                        help="Bootstrap resamples of the test set for the CI")
    parser.add_argument("--n_random_controls", type=int, default=5,
                        help="Random dimension subsets of the same size as the "
                             "E4 set, as a control for condition B")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    resolve_device(args.device)  # recorded for consistency; sklearn is CPU-only
    set_plot_style()

    reps_dir = Path(args.reps_dir)
    x_train, y_train = load_split(reps_dir, "train", args.encoder)
    x_test, y_test = load_split(reps_dir, "test", args.encoder)
    classes = sorted(set(y_train))
    print(f"  [data] train {x_train.shape}, test {x_test.shape}, {len(classes)} classes")
    for class_name in classes:
        print(f"    {class_name:20s} train {int((y_train == class_name).sum()):4d}  "
              f"test {int((y_test == class_name).sum()):4d}")

    run_dir = create_run_dir(
        EXPERIMENT, args.encoder, args.tag, args,
        extra={"classes": classes, "n_train": int(len(y_train)), "n_test": int(len(y_test))},
    )

    # ---- assemble the conditions -------------------------------------------
    conditions = {"A_full": (x_train, x_test, y_test)}

    if args.dimensions_file:
        spec = json.loads(Path(args.dimensions_file).read_text())
        if spec.get("h_dim") not in (None, x_train.shape[1]):
            raise ValueError(f"dimensions.json is for h_dim {spec['h_dim']}, "
                             f"representations are {x_train.shape[1]}")
        dims = np.asarray(spec["union"], dtype=int)
        print(f"  [E4] masking {len(dims)} dimensions from {args.dimensions_file}")

        masked_train, masked_test = x_train.copy(), x_test.copy()
        masked_train[:, dims] = 0.0
        masked_test[:, dims] = 0.0
        conditions["B_masked_e4"] = (masked_train, masked_test, y_test)

        # Control: same number of dimensions, chosen at random. Without this, a
        # drop in B could simply reflect losing len(dims) dimensions of anything.
        rng = np.random.default_rng(0)
        for i in range(args.n_random_controls):
            random_dims = rng.choice(x_train.shape[1], size=len(dims), replace=False)
            rand_train, rand_test = x_train.copy(), x_test.copy()
            rand_train[:, random_dims] = 0.0
            rand_test[:, random_dims] = 0.0
            conditions[f"B_masked_random_{i}"] = (rand_train, rand_test, y_test)
    else:
        print("  [E4] no --dimensions_file — condition B skipped")

    if args.transformed_reps:
        cache = torch.load(args.transformed_reps, map_location="cpu", weights_only=False)
        x_transformed = cache["reps"].float().numpy()
        y_transformed = np.asarray(
            cache.get("labels") or [Path(p).parent.name for p in cache["paths"]]
        )
        # Train on clean representations, test on transformed ones: this is the
        # deployment scenario, not a domain-adaptation one.
        conditions["C_transformed"] = (x_train, x_transformed, y_transformed)
    else:
        print("  [E3] no --transformed_reps — condition C skipped")

    # ---- fit -----------------------------------------------------------------
    rows, correct_masks, predictions = [], {}, {}
    for name, (train_features, test_features, test_labels) in conditions.items():
        for seed in range(args.n_seeds):
            set_seed(seed)
            y_pred, y_proba = fit_probe(train_features, y_train, test_features, seed)
            metrics = score(test_labels, y_pred, y_proba, classes)
            rows.append({"condition": name, "seed": seed, **metrics})
            if seed == 0:
                correct_masks[name] = (y_pred == test_labels)
                predictions[name] = (test_labels, y_pred, y_proba)
            print(f"  {name:24s} seed {seed}  "
                  f"bal_acc {metrics['balanced_accuracy']:.4f}  "
                  f"auc {metrics['macro_auc']:.4f}")

    save_metrics(run_dir, "metrics", rows,
                 ["condition", "seed", "balanced_accuracy", "macro_auc"])

    # ---- aggregate + significance -------------------------------------------
    summary = {}
    for name in conditions:
        values = np.array([r["balanced_accuracy"] for r in rows if r["condition"] == name])
        aucs = np.array([r["macro_auc"] for r in rows if r["condition"] == name])
        seed_mean, seed_lo, seed_hi = ci95(values)

        # The reported intervals come from resampling the test set, not from
        # refitting: see bootstrap_test_ci for why the seed spread is degenerate.
        test_labels, y_pred, y_proba = predictions[name]
        summary[name] = bootstrap_test_ci(test_labels, y_pred, y_proba, classes,
                                          args.n_boot, seed=0)
        summary[name]["seed_spread"] = {
            "balanced_accuracy_mean": seed_mean, "ci95": [seed_lo, seed_hi],
            "n_seeds": int(len(values)),
            "note": "zero-width when the solver is deterministic",
        }
        del aucs

    baseline = correct_masks.get("A_full")
    for name, mask in correct_masks.items():
        if name == "A_full" or baseline is None or len(mask) != len(baseline):
            continue
        summary[name]["mcnemar_vs_A_full"] = mcnemar(baseline, mask)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure -------------------------------------------------------------
    import matplotlib.pyplot as plt

    names = list(summary)
    means = [summary[n]["balanced_accuracy"]["mean"] for n in names]
    errors = np.array([
        [summary[n]["balanced_accuracy"]["mean"] - summary[n]["balanced_accuracy"]["ci95"][0]
         for n in names],
        [summary[n]["balanced_accuracy"]["ci95"][1] - summary[n]["balanced_accuracy"]["mean"]
         for n in names],
    ])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(names, means, yerr=errors, capsize=3, color="#4C72B0")
    ax.axhline(1.0 / len(classes), ls="--", lw=0.8, color="grey",
               label=f"chance ({1 / len(classes):.2f})")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"E5 — linear probe on frozen {args.encoder} representations\n"
                 f"error bars: 95% CI, stratified bootstrap over the test set "
                 f"(n={len(y_test)})")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.savefig(run_dir / "figures" / "probe_conditions.png")

    print("\n  " + json.dumps({k: v["balanced_accuracy"]["mean"] for k, v in summary.items()},
                              indent=2))
    print(f"Done — {run_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
