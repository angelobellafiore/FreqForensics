"""Plot ROC curves for FreqForensics evaluation results.

Loads the .npz file saved by scripts/evaluate.py and produces:
  - One overall ROC curve (all methods combined)
  - Four per-method ROC curves (each fake method vs real)
  - Random baseline diagonal
  - EER point on the overall curve
  - Optional 95% confidence band via bootstrap resampling (--bootstrap)

Usage:
    python visualisation/plot_roc.py \
        --results results/test_run.npz \
        --output  results/roc_curve.png

    # With confidence band (slower — 1000 bootstrap iterations):
    python visualisation/plot_roc.py \
        --results   results/test_run.npz \
        --output    results/roc_curve_ci.png \
        --bootstrap --n_bootstrap 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, roc_auc_score


METHOD_COLORS = {
    'Deepfakes':      '#e41a1c',
    'Face2Face':      '#377eb8',
    'FaceSwap':       '#4daf4a',
    'NeuralTextures': '#ff7f00',
}

# Common FPR grid for interpolating bootstrap curves
_FPR_GRID = np.linspace(0, 1, 500)


def _compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> tuple[float, float, float]:
    """Return (eer, fpr_at_eer, tpr_at_eer)."""
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(fpr[idx]), float(tpr[idx])


def _bootstrap_ci(
    labels:      np.ndarray,
    probs:       np.ndarray,
    n_bootstrap: int,
    rng:         np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (tpr_lower, tpr_upper) at each point of _FPR_GRID.

    For each bootstrap resample we interpolate its ROC curve onto the common
    FPR grid, then take the 2.5th and 97.5th percentiles across resamples.
    """
    n = len(labels)
    tpr_matrix = np.zeros((n_bootstrap, len(_FPR_GRID)))

    for i in range(n_bootstrap):
        idx          = rng.integers(0, n, size=n)   # resample with replacement
        fpr_i, tpr_i, _ = roc_curve(labels[idx], probs[idx])
        # Interpolate onto the common grid (monotone, so np.interp works)
        tpr_matrix[i] = np.interp(_FPR_GRID, fpr_i, tpr_i)

    tpr_lower = np.percentile(tpr_matrix, 2.5,  axis=0)
    tpr_upper = np.percentile(tpr_matrix, 97.5, axis=0)
    return tpr_lower, tpr_upper


def plot_roc(
    results_path: Path,
    output_path:  Path,
    bootstrap:    bool = False,
    n_bootstrap:  int  = 1000,
    seed:         int  = 42,
) -> None:
    data    = np.load(results_path, allow_pickle=True)
    probs   = data['probs']
    labels  = data['labels']
    methods = data['methods']

    fig, ax = plt.subplots(figsize=(7, 6))

    # --- Random baseline ---
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    # --- Per-method curves ---
    real_mask         = labels == 0
    fake_method_names = [m for m in METHOD_COLORS if m in set(methods)]

    for method in fake_method_names:
        fake_mask = methods == method
        combined  = fake_mask | real_mask
        y_true    = labels[combined]
        y_score   = probs[combined]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc         = roc_auc_score(y_true, y_score)

        ax.plot(
            fpr, tpr,
            color=METHOD_COLORS[method],
            lw=1.2, alpha=0.7,
            label=f'{method} (AUC = {auc:.3f})',
        )

    # --- Overall curve ---
    fpr_all, tpr_all, _ = roc_curve(labels, probs)
    auc_all = roc_auc_score(labels, probs)
    eer, eer_fpr, eer_tpr = _compute_eer(fpr_all, tpr_all)

    # --- Bootstrap confidence band (overall curve only) ---
    if bootstrap:
        print(f"  Computing 95% CI with {n_bootstrap} bootstrap iterations...")
        rng = np.random.default_rng(seed)
        tpr_lower, tpr_upper = _bootstrap_ci(labels, probs, n_bootstrap, rng)
        ax.fill_between(
            _FPR_GRID, tpr_lower, tpr_upper,
            color='black', alpha=0.15,
            label='95% CI (bootstrap)',
        )

    ax.plot(
        fpr_all, tpr_all,
        color='black', lw=2.5,
        label=f'Overall (AUC = {auc_all:.3f})',
    )

    # --- EER point ---
    ax.scatter(
        [eer_fpr], [eer_tpr],
        color='black', s=60, zorder=5,
        label=f'EER = {eer:.3f}',
    )
    ax.annotate(
        f'EER={eer:.3f}',
        xy=(eer_fpr, eer_tpr),
        xytext=(eer_fpr + 0.05, eer_tpr - 0.07),
        fontsize=8,
        arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
    )

    # --- Formatting ---
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    title = 'ROC Curves — FreqForensics on FF++ Test Set'
    if bootstrap:
        title += ' (95% CI)'
    ax.set_title(title, fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved ROC curve to: {output_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--results',     type=Path, default=Path('results/test_run.npz'))
    p.add_argument('--output',      type=Path, default=Path('results/roc_curve.png'))
    p.add_argument('--bootstrap',   action='store_true',
                   help='Add 95%% confidence band via bootstrap resampling')
    p.add_argument('--n_bootstrap', type=int, default=1000,
                   help='Number of bootstrap iterations (default: 1000)')
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()

    plot_roc(
        args.results, args.output,
        bootstrap=args.bootstrap,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
