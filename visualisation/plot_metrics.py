"""Plot per-method AUC bar chart and score distribution for FreqForensics.

Loads the .npz file saved by scripts/evaluate.py and produces:
  - Per-method AUC horizontal bar chart with optional 95% CI error bars
  - Score distribution histogram (real vs fake predicted probabilities)

Usage:
    python visualisation/plot_metrics.py \
        --results results/test_run.npz \
        --output_dir results/

    # With bootstrap confidence intervals on the bar chart:
    python visualisation/plot_metrics.py \
        --results results/test_run.npz \
        --output_dir results/ \
        --bootstrap --n_bootstrap 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score


METHOD_COLORS = {
    'Deepfakes':      '#e41a1c',
    'Face2Face':      '#377eb8',
    'FaceSwap':       '#4daf4a',
    'NeuralTextures': '#ff7f00',
}


def _bootstrap_auc_ci(
    labels:      np.ndarray,
    probs:       np.ndarray,
    n_bootstrap: int,
    rng:         np.random.Generator,
) -> tuple[float, float]:
    """Return (ci_lower, ci_upper) for AUC via bootstrap resampling."""
    n    = len(labels)
    aucs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        # Skip resamples with only one class (rare but possible with small subsets)
        if len(np.unique(labels[idx])) < 2:
            aucs[i] = np.nan
        else:
            aucs[i] = roc_auc_score(labels[idx], probs[idx])
    aucs = aucs[~np.isnan(aucs)]
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def plot_per_method_auc(
    probs:       np.ndarray,
    labels:      np.ndarray,
    methods:     np.ndarray,
    output_path: Path,
    bootstrap:   bool = False,
    n_bootstrap: int  = 1000,
    seed:        int  = 42,
) -> None:
    """Horizontal bar chart of per-method AUC with optional 95% CI error bars."""
    real_mask         = labels == 0
    fake_method_names = [m for m in METHOD_COLORS if m in set(methods)]
    rng               = np.random.default_rng(seed)

    method_aucs: dict[str, float]               = {}
    method_ci:   dict[str, tuple[float, float]] = {}

    for method in fake_method_names:
        fake_mask = methods == method
        combined  = fake_mask | real_mask
        y_true    = labels[combined]
        y_score   = probs[combined]

        method_aucs[method] = roc_auc_score(y_true, y_score)

        if bootstrap:
            method_ci[method] = _bootstrap_auc_ci(y_true, y_score, n_bootstrap, rng)

    overall_auc = roc_auc_score(labels, probs)

    # Sort ascending so best method is at top
    sorted_methods = sorted(method_aucs, key=method_aucs.get)
    sorted_aucs    = [method_aucs[m] for m in sorted_methods]
    colors         = [METHOD_COLORS[m] for m in sorted_methods]

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.barh(sorted_methods, sorted_aucs, color=colors, height=0.5, alpha=0.85)

    # Bootstrap CI error bars
    if bootstrap:
        for i, method in enumerate(sorted_methods):
            ci_lo, ci_hi = method_ci[method]
            auc          = method_aucs[method]
            ax.errorbar(
                auc, i,
                xerr=[[auc - ci_lo], [ci_hi - auc]],
                fmt='none',
                color='black', capsize=5, capthick=1.5, elinewidth=1.5,
                zorder=5,
            )

    # Annotate each bar with its AUC value — placed at 60% along the visible
    # bar length so it stays clear of the error bars at the right end
    x_start = 0.85   # matches ax.set_xlim left bound
    for bar, auc in zip(bars, sorted_aucs):
        x_label = x_start + (auc - x_start) * 0.6
        ax.text(
            x_label, bar.get_y() + bar.get_height() / 2,
            f'{auc:.3f}',
            va='center', ha='center', fontsize=10, color='white', fontweight='bold',
        )

    # Overall AUC reference line
    ax.axvline(overall_auc, color='black', lw=1.5, linestyle='--',
               label=f'Overall AUC = {overall_auc:.3f}')

    ax.set_xlim([0.85, 1.0])
    ax.set_xlabel('AUC-ROC', fontsize=12)
    title = 'Per-Method AUC — FreqForensics on FF++ Test Set'
    if bootstrap:
        title += ' (95% CI)'
    ax.set_title(title, fontsize=13)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved per-method bar chart to: {output_path}")
    plt.close(fig)


def plot_score_distribution(
    probs:       np.ndarray,
    labels:      np.ndarray,
    output_path: Path,
) -> None:
    """Histogram of predicted probabilities for real vs fake samples."""
    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]

    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(0, 1, 51)

    ax.hist(real_probs, bins=bins, alpha=0.6, color='#377eb8',
            label='Real', density=True)
    ax.hist(fake_probs, bins=bins, alpha=0.6, color='#e41a1c',
            label='Fake', density=True)

    ax.axvline(0.5, color='black', lw=1.5, linestyle='--', label='Threshold = 0.5')

    ax.set_xlabel('Predicted Probability of Being Fake', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution — Real vs Fake', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved score distribution to: {output_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--results',     type=Path, default=Path('results/test_run.npz'))
    p.add_argument('--output_dir',  type=Path, default=Path('results'))
    p.add_argument('--bootstrap',   action='store_true',
                   help='Add 95%% CI error bars to the per-method bar chart')
    p.add_argument('--n_bootstrap', type=int, default=1000)
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()

    data    = np.load(args.results, allow_pickle=True)
    probs   = data['probs']
    labels  = data['labels']
    methods = data['methods']

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.bootstrap:
        print(f"  Computing 95% CI with {args.n_bootstrap} bootstrap iterations per method...")

    plot_per_method_auc(
        probs, labels, methods,
        args.output_dir / 'per_method_auc.png',
        bootstrap=args.bootstrap,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    plot_score_distribution(probs, labels,
                            args.output_dir / 'score_distribution.png')


if __name__ == '__main__':
    main()