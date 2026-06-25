"""Plot ablation study results for FreqForensics.

Produces two charts:
  1. Overall AUC comparison across ablation variants (grouped bar)
  2. Per-method AUC comparison: full model vs spatial_only

Usage:
    python visualisation/plot_ablation.py \
        --ablation  results/ablation.npz \
        --output_dir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


VARIANT_LABELS = {
    'full':         'Full model',
    'no_lf':        'No LF branch',
    'no_hf':        'No HF branch',
    'spatial_only': 'Spatial only',
}

VARIANT_COLORS = {
    'full':         '#2166ac',
    'no_lf':        '#f4a582',
    'no_hf':        '#d6604d',
    'spatial_only': '#b2182b',
}

METHOD_COLORS = {
    'Deepfakes':      '#e41a1c',
    'Face2Face':      '#377eb8',
    'FaceSwap':       '#4daf4a',
    'NeuralTextures': '#ff7f00',
}

# Per-method AUC values from the ablation run — hardcoded from output
PER_METHOD = {
    'full': {
        'Deepfakes': 0.9489, 'Face2Face': 0.9294,
        'FaceSwap':  0.9146, 'NeuralTextures': 0.9098,
    },
    'spatial_only': {
        'Deepfakes': 0.9489, 'Face2Face': 0.9415,
        'FaceSwap':  0.9035, 'NeuralTextures': 0.8972,
    },
}


def plot_overall_ablation(
    ablation_path: Path,
    output_path:   Path,
) -> None:
    """Horizontal bar chart comparing overall AUC across ablation variants."""
    data = np.load(ablation_path, allow_pickle=True)

    variants = list(VARIANT_LABELS.keys())
    aucs     = [float(data[v]) for v in variants]
    labels   = [VARIANT_LABELS[v] for v in variants]
    colors   = [VARIANT_COLORS[v] for v in variants]
    full_auc = float(data['full'])

    fig, ax = plt.subplots(figsize=(7, 3.5))

    bars = ax.barh(labels, aucs, color=colors, height=0.5, alpha=0.88)

    # Annotate with AUC and drop
    x_start = 0.90
    for bar, auc, variant in zip(bars, aucs, variants):
        x_label = x_start + (auc - x_start) * 0.5
        ax.text(
            x_label, bar.get_y() + bar.get_height() / 2,
            f'{auc:.4f}',
            va='center', ha='center', fontsize=9, color='white', fontweight='bold',
        )
        if variant != 'full':
            drop = full_auc - auc
            ax.text(
                auc + 0.0005, bar.get_y() + bar.get_height() / 2,
                f'−{drop:.4f}',
                va='center', ha='left', fontsize=8, color='#555555',
            )

    # Full model reference line
    ax.axvline(full_auc, color='#2166ac', lw=1.5, linestyle='--',
               label=f'Full model = {full_auc:.4f}')

    ax.set_xlim([0.90, 0.945])
    ax.set_xlabel('AUC-ROC', fontsize=11)
    ax.set_title('Ablation Study — Overall AUC on FF++ Test Set', fontsize=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved overall ablation chart to: {output_path}")
    plt.close(fig)


def plot_per_method_ablation(output_path: Path) -> None:
    """Grouped bar chart: full model vs spatial_only per fake method."""
    methods  = list(METHOD_COLORS.keys())
    full_auc = [PER_METHOD['full'][m]         for m in methods]
    spa_auc  = [PER_METHOD['spatial_only'][m] for m in methods]

    x     = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))

    bars_full = ax.bar(x - width / 2, full_auc, width, label='Full model',
                       color='#2166ac', alpha=0.88)
    bars_spa  = ax.bar(x + width / 2, spa_auc,  width, label='Spatial only',
                       color='#b2182b', alpha=0.88)

    # Annotate bars
    for bar in list(bars_full) + list(bars_spa):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.004,
            f'{bar.get_height():.3f}',
            ha='center', va='top', fontsize=8, color='white', fontweight='bold',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim([0.87, 0.97])
    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_title('Per-Method AUC: Full Model vs Spatial Only', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved per-method ablation chart to: {output_path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--ablation',    type=Path, default=Path('results/ablation.npz'))
    p.add_argument('--output_dir',  type=Path, default=Path('results'))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_overall_ablation(
        args.ablation,
        args.output_dir / 'ablation_overall.png',
    )
    plot_per_method_ablation(
        args.output_dir / 'ablation_per_method.png',
    )


if __name__ == '__main__':
    main()
