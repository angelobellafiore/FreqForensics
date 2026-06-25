"""Inference-time ablation study for FreqForensics.

Evaluates branch ablations by zeroing out specific branch feature vectors
before fusion. This approximates the contribution of each branch without
requiring full retraining.

Ablation variants:
  full          — all three branches active (baseline)
  spatial_only  — LF and HF branches zeroed out
  no_lf         — LF branch zeroed out
  no_hf         — HF branch zeroed out

Note: loss-term ablations (no Fo-Mixup, no L_local, no L_global) require
retraining and are not included here.

Usage:
    python scripts/run_ablation.py \
        --checkpoint best.pt \
        --crops_csv  data/index_with_crops.csv \
        [--crops_root /path/to/crops] \
        --output     results/ablation.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))

from models.freqforensics import FreqForensics
from models.freq_transforms import build_lf_tensor, build_hf_tensor
from data.dataset import FFPPDataset
from evaluation.metrics import evaluate, print_report


# ---------------------------------------------------------------------------
# Ablated model wrapper
# ---------------------------------------------------------------------------

class AblatedFreqForensics(nn.Module):
    """Wraps FreqForensics and zeros out specified branch outputs before fusion.

    Args:
        model:       Loaded FreqForensics instance
        zero_lf:     If True, replace LF branch output with zeros
        zero_hf:     If True, replace HF branch output with zeros
    """

    def __init__(
        self,
        model:   FreqForensics,
        zero_lf: bool = False,
        zero_hf: bool = False,
    ) -> None:
        super().__init__()
        self.model   = model
        self.zero_lf = zero_lf
        self.zero_hf = zero_hf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_s  = self.model.spatial_branch(x)
        f_lf = self.model.lf_encoder(build_lf_tensor(x))
        f_hf = self.model.hf_encoder(build_hf_tensor(x))

        if self.zero_lf:
            f_lf = torch.zeros_like(f_lf)
        if self.zero_hf:
            f_hf = torch.zeros_like(f_hf)

        fused = self.model.fusion(f_s, f_lf, f_hf)
        return self.model.head(fused)


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

ABLATIONS: dict[str, dict] = {
    'full':         {'zero_lf': False, 'zero_hf': False},
    'no_lf':        {'zero_lf': True,  'zero_hf': False},
    'no_hf':        {'zero_lf': False, 'zero_hf': True},
    'spatial_only': {'zero_lf': True,  'zero_hf': True},
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Inference-time ablation study.')
    p.add_argument('--checkpoint',  required=True, type=Path)
    p.add_argument('--crops_csv',   required=True, type=Path)
    p.add_argument('--crops_root',  default=None,  type=Path,
                   help='Rewrite crop paths to this root (use on Colab/Kaggle)')
    p.add_argument('--split',       default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--batch_size',  default=64,  type=int)
    p.add_argument('--num_workers', default=4,   type=int)
    p.add_argument('--output',      default=Path('results/ablation.npz'), type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # --- Load base model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model = FreqForensics().to(device)
    base_model.load_state_dict(state['model'])
    base_model.eval()
    print(f"  Checkpoint epoch: {state.get('epoch', '?')}  "
          f"best_auc: {state.get('best_auc', '?')}\n")

    # --- Load data ---
    df = pd.read_csv(args.crops_csv)
    if 'crop_path' in df.columns:
        df['path'] = df['crop_path'].where(
            df['crop_path'].notna() & (df['crop_path'] != ''), df['path']
        )
    if args.crops_root is not None:
        df['path'] = df['path'].apply(
            lambda p: str(args.crops_root / Path(*Path(p).parts[-3:]))
        )
    split_df = df[df['split'] == args.split].reset_index(drop=True)
    print(f"{args.split} split: {len(split_df):,} frames\n")

    loader = DataLoader(
        FFPPDataset(split_df, is_train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # --- Run each ablation variant ---
    results: dict[str, float] = {}

    for name, kwargs in ABLATIONS.items():
        print(f"{'─' * 40}")
        print(f"Variant: {name}")
        ablated = AblatedFreqForensics(base_model, **kwargs).to(device)
        result  = evaluate(ablated, loader, device)
        print_report(result, split=f'{args.split} [{name}]')
        results[name] = result.auc

    # --- Summary table ---
    print(f"\n{'=' * 40}")
    print(f"  Ablation summary — AUC-ROC ({args.split})")
    print(f"{'=' * 40}")
    full_auc = results['full']
    for name, auc in results.items():
        drop = full_auc - auc
        marker = '' if name == 'full' else f'  (Δ = -{drop:.4f})'
        print(f"  {name:<16s}  {auc:.4f}{marker}")
    print(f"{'=' * 40}\n")

    # --- Save results ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{k: np.array(v) for k, v in results.items()})
    print(f"Ablation results saved to: {args.output}")


if __name__ == '__main__':
    main()
