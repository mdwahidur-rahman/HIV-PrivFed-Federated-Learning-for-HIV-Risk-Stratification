"""
Statistical evaluation utilities:
  - Youden's J threshold selection on TRAINING predictions only (Eq. youden),
    never on the test set.
  - Non-parametric bootstrap confidence intervals (Section: Statistical
    Robustness; 1,000 resamples, 95% CI).
  - Paired bootstrap significance testing between two models evaluated on the
    SAME test set (Table 9).

Review issue #7 notes that confidence intervals were reported for only two
conventional models, while the federated/private configurations central to
the paper's contribution had none. The functions here are written to be
called on ANY model's predictions -- there is nothing conventional-ML-specific
about them -- specifically so that gap is mechanical to close once you have
saved prediction arrays for the federated and private models too.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, matthews_corrcoef

from .config import EVAL


def youden_threshold(y_true_train: np.ndarray, y_prob_train: np.ndarray) -> float:
    """
    tau* = argmax_tau [TPR(tau) - FPR(tau)]  (Eq. youden), computed on POOLED
    TRAINING predictions only. Never call this with test-set labels/probs.
    """
    fpr, tpr, thresholds = roc_curve(y_true_train, y_prob_train)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


@dataclass
class BootstrapCI:
    point_estimate: float
    mean: float
    ci_low: float
    ci_high: float
    n_resamples: int


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = EVAL.n_bootstrap,
    ci_level: float = EVAL.ci_level,
    random_seed: int = EVAL.random_seed,
) -> BootstrapCI:
    """
    Resamples (y_true, y_prob) pairs WITH replacement `n_resamples` times,
    recomputes `metric_fn` each time, and returns the point estimate (on the
    original, unresampled data) alongside the bootstrap mean and percentile CI.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    point_estimate = metric_fn(y_true, y_prob)

    values = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            values[b] = np.nan  # AUROC/AUPRC undefined for a single-class resample
            continue
        values[b] = metric_fn(yt, yp)

    values = values[~np.isnan(values)]
    alpha = 1 - ci_level
    lo = np.percentile(values, 100 * (alpha / 2))
    hi = np.percentile(values, 100 * (1 - alpha / 2))
    return BootstrapCI(
        point_estimate=point_estimate, mean=float(values.mean()),
        ci_low=float(lo), ci_high=float(hi), n_resamples=len(values),
    )


@dataclass
class PairedBootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    significant_at_alpha: bool


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = EVAL.n_bootstrap,
    alpha: float = 1 - EVAL.ci_level,
    random_seed: int = EVAL.random_seed,
) -> PairedBootstrapResult:
    """
    Paired bootstrap: resample the SAME indices for both models' predictions
    on every iteration (so the comparison is paired, not independent), and
    report the CI on (metric_a - metric_b). Significant iff the CI excludes 0.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    diffs = np.empty(n_resamples, dtype=float)

    for b in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            diffs[b] = np.nan
            continue
        m_a = metric_fn(yt, y_prob_a[idx])
        m_b = metric_fn(yt, y_prob_b[idx])
        diffs[b] = m_a - m_b

    diffs = diffs[~np.isnan(diffs)]
    lo = np.percentile(diffs, 100 * (alpha / 2))
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    point_diff = metric_fn(y_true, y_prob_a) - metric_fn(y_true, y_prob_b)
    significant = not (lo <= 0 <= hi)
    return PairedBootstrapResult(mean_diff=point_diff, ci_low=float(lo), ci_high=float(hi), significant_at_alpha=significant)


def multi_seed_summary(values_by_seed: Dict[int, float]) -> Dict[str, float]:
    """
    Summarize a metric computed across multiple independent TRAINING seeds
    (not test-resampling) -- distinct from bootstrap_ci, which quantifies
    sampling variance in the fixed test set for a SINGLE trained model.
    Review issue #7 specifically asks these two sources of variance
    (retraining vs. resampling) not be conflated; this function is the
    retraining-variance counterpart to bootstrap_ci above.
    """
    vals = np.array(list(values_by_seed.values()), dtype=float)
    return {
        "n_seeds": len(vals),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


# Convenience metric functions with the (y_true, y_prob) signature bootstrap_ci expects
def metric_auroc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)


def metric_auprc(y_true, y_prob):
    return average_precision_score(y_true, y_prob)


def metric_mcc_at_threshold(threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
    def _fn(y_true, y_prob):
        y_pred = (y_prob >= threshold).astype(int)
        return matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    return _fn
