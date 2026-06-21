"""Evaluate a FreqForensics checkpoint on the test (or val) split.

Loads a saved checkpoint, runs inference on the requested split, prints a
full metrics report, and optionally saves raw logits/labels to results/.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --crops_csv  data/index_with_crops.csv \
        [--split test] \
        [--batch_size 64] \
        [--save_results results/test_run.npz]

Colab:
    python scripts/evaluate.py \
        --checkpoint /content/drive/MyDrive/FreqForensics/checkpoints/best.pt \
        --crops_csv  /content/drive/MyDrive/FreqForensics/index_with_crops.csv \
        --save_results /content/drive/MyDrive/FreqForensics/results/test_run.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))

from models.freqforensics import FreqForensics
from data.dataset import FFPPDataset
from evaluation.metrics import evaluate, print_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate a FreqForensics checkpoint.')
    p.add_argument('--checkpoint',   required=True, type=Path,
                   help='Path to .pt checkpoint file (best.pt or latest.pt)')
    p.add_argument('--crops_csv',    required=True, type=Path,
                   help='Index CSV with crop_path and split columns')
    p.add_argument('--split',        default='test', choices=['train', 'val', 'test'],
                   help='Which split to evaluate (default: test)')
    p.add_argument('--batch_size',   default=64,  type=int)
    p.add_argument('--num_workers',  default=4,   type=int)
    p.add_argument('--save_results', default=None, type=Path,
                   help='Optional path to save raw logits/labels as .npz')
    return p.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> FreqForensics:
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = FreqForensics().to(device)
    model.load_state_dict(state['model'])
    model.eval()

    epoch = state.get('epoch', '?')
    auc   = state.get('best_auc', '?')
    print(f"  Checkpoint epoch: {epoch}  |  best_auc (at save time): {auc}")
    return model


def build_loader(
    crops_csv:   Path,
    split:       str,
    batch_size:  int,
    num_workers: int,
    device:      torch.device,
) -> DataLoader:
    df = pd.read_csv(crops_csv)

    # Use crop_path if available, otherwise fall back to path
    if 'crop_path' in df.columns:
        df['path'] = df['crop_path'].where(
            df['crop_path'].notna() & (df['crop_path'] != ''),
            df['path']
        )

    split_df = df[df['split'] == split].reset_index(drop=True)
    print(f"  {split} split: {len(split_df):,} frames")

    dataset = FFPPDataset(split_df, is_train=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}\n")

    model  = load_model(args.checkpoint, device)
    loader = build_loader(
        args.crops_csv, args.split,
        args.batch_size, args.num_workers, device,
    )

    print(f"\nRunning inference on '{args.split}' split...")
    result = evaluate(model, loader, device)
    print_report(result, split=args.split)

    if args.save_results is not None:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_results,
            logits=result.logits,
            probs=result.probs,
            labels=result.labels,
            methods=np.array(result.methods),
        )
        print(f"Raw results saved to: {args.save_results}")


if __name__ == '__main__':
    main()
