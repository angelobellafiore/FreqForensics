"""Training configuration for FreqForensics.

All hyperparameters live here. Pass a TrainingConfig instance to Trainer —
never hardcode values inside the training loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    # --- Paths ---
    crops_csv:  Path = Path('data/index_with_crops.csv')
    output_dir: Path = Path('checkpoints')
    # If set, rewrites the crop_path prefix to this directory.
    # Use on Colab when the CSV was built on a different machine:
    #   --crops_root /content/drive/MyDrive/FreqForensics/crops
    crops_root: Path | None = None

    # --- Data ---
    batch_size:  int = 32
    num_workers: int = 4
    image_size:  int = 224

    # --- Optimiser ---
    lr:            float = 1e-4
    weight_decay:  float = 1e-4
    epochs:        int   = 20
    warmup_epochs: int   = 2

    # --- LR scheduler (cosine annealing) ---
    lr_min: float = 1e-6

    # --- Loss weights ---
    lambda_aux:  float = 0.1
    beta_local:  float = 0.5
    beta_global: float = 0.5
    cam_every_n: int   = 10   # compute L_local every N steps; 0 to disable

    # --- Fo-Mixup ---
    fo_mixup_alpha: float = 0.5   # Beta(alpha, alpha) controls blend strength

    # --- Logging / checkpointing ---
    seed:               int = 42
    log_every_n:        int = 50    # log loss breakdown every N steps
    val_every_n_epochs: int = 1     # run validation every N epochs
    save_every_n:       int = 500   # save latest.pt every N steps (mid-epoch safety)

    # --- Ablation ---
    spatial_only: bool = False  # train without LF/HF branches (retrained ablation)

    # --- Runtime (set automatically, not by user) ---
    device: str = field(default='cpu', repr=True)

    # ------------------------------------------------------------------
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> TrainingConfig:
        """Build a config from argparse output, overriding only supplied args."""
        cfg = cls()
        for key, value in vars(args).items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)
        # Coerce Path fields
        cfg.crops_csv  = Path(cfg.crops_csv)
        cfg.output_dir = Path(cfg.output_dir)
        if cfg.crops_root is not None:
            cfg.crops_root = Path(cfg.crops_root)
        return cfg

    def __post_init__(self) -> None:
        self.crops_csv  = Path(self.crops_csv)
        self.output_dir = Path(self.output_dir)
        if self.crops_root is not None:
            self.crops_root = Path(self.crops_root)


if __name__ == '__main__':
    cfg = TrainingConfig()
    print("Default TrainingConfig:")
    for f_name, f_val in cfg.__dataclass_fields__.items():
        print(f"  {f_name:20s} = {getattr(cfg, f_name)}")
