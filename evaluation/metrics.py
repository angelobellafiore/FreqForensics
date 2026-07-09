"""Evaluation metrics for FreqForensics.

Computes AUC-ROC, accuracy, EER, Average Precision, and per-method AUC
from model predictions on a DataLoader.

Usage:
    from evaluation.metrics import evaluate, print_report

    result = evaluate(model, loader, device)
    print_report(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


@dataclass
class MetricsResult:
    """Container for all evaluation outputs."""

    # Overall metrics
    auc:      float = 0.0
    acc:      float = 0.0
    eer:      float = 0.0
    ap:       float = 0.0

    # Per-method AUC  {method_name: auc_score}
    per_method_auc: dict[str, float] = field(default_factory=dict)

    # Raw arrays, kept for plotting ROC curves, calibration, etc.
    logits:  np.ndarray = field(default_factory=lambda: np.array([]))
    probs:   np.ndarray = field(default_factory=lambda: np.array([]))
    labels:  np.ndarray = field(default_factory=lambda: np.array([]))
    methods: list[str]  = field(default_factory=list)


def _compute_eer(labels: np.ndarray, probs: np.ndarray) -> float:
    """Equal Error Rate: the threshold where FPR == FNR."""
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    # Find the index where |FPR - FNR| is minimised
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


@torch.no_grad()
def evaluate(
    model:  torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> MetricsResult:
    """Run inference on loader and compute all metrics.

    Args:
        model:  FreqForensics (or any model with a forward() -> (B,1) logit)
        loader: DataLoader returning (images, labels, video_ids, methods)
        device: torch device

    Returns:
        MetricsResult with all computed metrics.
    """
    model.eval()
    device = torch.device(device)

    all_logits:  list[torch.Tensor] = []
    all_labels:  list[torch.Tensor] = []
    all_methods: list[str]          = []

    for batch in loader:
        images, labels, _video_ids, methods = batch
        images = images.to(device)

        logit = model(images)                        # (B, 1)
        all_logits.append(logit.squeeze(1).cpu())
        all_labels.append(labels)
        all_methods.extend(methods)

    logits  = torch.cat(all_logits).numpy()
    labels  = torch.cat(all_labels).numpy()
    probs   = 1.0 / (1.0 + np.exp(-logits))         # sigmoid

    # --- Overall metrics ---
    auc = float(roc_auc_score(labels, probs))
    ap  = float(average_precision_score(labels, probs))
    eer = _compute_eer(labels, probs)
    acc = float(((probs >= 0.5).astype(int) == labels).mean())

    # --- Per-method AUC ---
    # Each fake method only has label=1, so we pair it with all real samples
    # (label=0) to compute a meaningful binary AUC per method.
    methods_arr   = np.array(all_methods)
    real_mask     = labels == 0
    fake_methods  = sorted(m for m in set(all_methods) if m != 'original')
    per_method_auc: dict[str, float] = {}

    for method in fake_methods:
        fake_mask = methods_arr == method
        combined  = fake_mask | real_mask
        if combined.sum() < 2:
            continue
        per_method_auc[method] = float(
            roc_auc_score(labels[combined], probs[combined])
        )

    return MetricsResult(
        auc=auc,
        acc=acc,
        eer=eer,
        ap=ap,
        per_method_auc=per_method_auc,
        logits=logits,
        probs=probs,
        labels=labels,
        methods=all_methods,
    )


def print_report(result: MetricsResult, split: str = 'test') -> None:
    """Print a formatted evaluation report to stdout."""
    print(f"\n{'=' * 50}")
    print(f"  Evaluation results, {split}")
    print(f"{'=' * 50}")
    print(f"  AUC-ROC  : {result.auc:.4f}")
    print(f"  Acc@0.5  : {result.acc:.4f}")
    print(f"  EER      : {result.eer:.4f}")
    print(f"  Avg Prec : {result.ap:.4f}")

    if result.per_method_auc:
        print(f"\n  Per-method AUC:")
        for method, auc in sorted(result.per_method_auc.items()):
            print(f"    {method:<20s} {auc:.4f}")

    print(f"{'=' * 50}\n")


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

    print("Running smoke test for metrics.py...")

    import torch.nn as nn

    # Minimal stub model that returns random logits
    class _DummyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], 1)

    from torch.utils.data import DataLoader, TensorDataset

    B = 200
    images  = torch.rand(B, 3, 224, 224)
    labels  = torch.randint(0, 2, (B,))
    # Simulate method strings, DataLoader returns tuples so we need a custom collate
    methods = ['Deepfakes'] * 50 + ['Face2Face'] * 50 + ['FaceSwap'] * 50 + ['original'] * 50

    # Package into a list-of-tuples dataset
    class _SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.images  = images
            self.labels  = labels
            self.methods = methods

        def __len__(self):
            return B

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx], 'vid_000', self.methods[idx]

    loader = DataLoader(_SyntheticDataset(), batch_size=32)
    model  = _DummyModel()
    device = torch.device('cpu')

    result = evaluate(model, loader, device)

    # With a random model AUC should be ~0.5
    assert 0.0 <= result.auc <= 1.0,  f"AUC out of range: {result.auc}"
    assert 0.0 <= result.eer <= 1.0,  f"EER out of range: {result.eer}"
    assert 0.0 <= result.ap  <= 1.0,  f"AP out of range:  {result.ap}"
    assert len(result.per_method_auc) > 0, "No per-method AUC computed"

    print_report(result, split='smoke-test')
    print("All assertions passed.")
