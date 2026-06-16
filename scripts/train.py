"""Entry point for FreqForensics training.

Usage (local):
    python scripts/train.py \
        --crops_csv data/index_with_crops.csv \
        --output_dir checkpoints \
        --epochs 20 \
        --batch_size 32

Usage (Colab / GPU):
    python scripts/train.py \
        --crops_csv /content/drive/MyDrive/FreqForensics/index_with_crops.csv \
        --output_dir /content/drive/MyDrive/FreqForensics/checkpoints \
        --epochs 20 \
        --batch_size 32 \
        --num_workers 2
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Allow sibling-directory imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import TrainingConfig
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train FreqForensics deepfake detector.')

    # Paths
    p.add_argument('--crops_csv',  type=Path, default=None)
    p.add_argument('--output_dir', type=Path, default=None)
    p.add_argument('--crops_root', type=Path, default=None,
                   help='Rewrite crop paths to this root (use on Colab)')

    # Data
    p.add_argument('--batch_size',  type=int,   default=None)
    p.add_argument('--num_workers', type=int,   default=None)

    # Optimiser
    p.add_argument('--lr',            type=float, default=None)
    p.add_argument('--weight_decay',  type=float, default=None)
    p.add_argument('--epochs',        type=int,   default=None)
    p.add_argument('--warmup_epochs', type=int,   default=None)

    # Loss
    p.add_argument('--lambda_aux',  type=float, default=None)
    p.add_argument('--beta_local',  type=float, default=None)
    p.add_argument('--beta_global', type=float, default=None)
    p.add_argument('--cam_every_n', type=int,   default=None)

    # Misc
    p.add_argument('--seed', type=int, default=None)

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg  = TrainingConfig.from_args(args)

    # Detect device
    if torch.cuda.is_available():
        cfg.device = 'cuda'
    elif torch.backends.mps.is_available():
        cfg.device = 'mps'
    else:
        cfg.device = 'cpu'

    set_seed(cfg.seed)

    print("=" * 60)
    print("FreqForensics — Training")
    print("=" * 60)
    print(f"  device      : {cfg.device}")
    print(f"  crops_csv   : {cfg.crops_csv}")
    print(f"  output_dir  : {cfg.output_dir}")
    print(f"  epochs      : {cfg.epochs}")
    print(f"  batch_size  : {cfg.batch_size}")
    print(f"  lr          : {cfg.lr}")
    print(f"  seed        : {cfg.seed}")
    print("=" * 60)

    trainer = Trainer(cfg)
    trainer.train()

    print("\nTraining complete.")
    print(f"Best val AUC: {trainer.best_auc:.4f}")
    print(f"Checkpoints saved to: {cfg.output_dir}")


if __name__ == '__main__':
    main()
