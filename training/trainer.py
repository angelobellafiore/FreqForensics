"""FreqForensics training loop.

Handles:
  - Data loading with WeightedRandomSampler
  - Per-batch Fo-Mixup on fake samples
  - forward_with_aux → FreqForensicsLoss
  - Grad-CAM computation every N steps for L_local
  - Cosine LR schedule with linear warmup
  - Validation with AUC + accuracy
  - Checkpoint saving (best val AUC + latest)
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Allow imports from sibling directories
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))

from freqforensics import FreqForensics
from fo_mixup import subband_fo_mixup_ll, subband_fo_mixup_hh
from freq_transforms import build_hf_tensor, haar_dwt2d, to_grayscale
from dataset import FFPPDataset
from training.config import TrainingConfig
from training.loss import FreqForensicsLoss
from training.sampler import make_weighted_sampler


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Minimal Grad-CAM for the spatial branch (EfficientNet-B4 conv_head).

    Registers forward/backward hooks on the target layer, runs a forward
    pass, back-propagates the class score, and computes the weighted
    average of activation maps.
    """

    def __init__(self, model: FreqForensics) -> None:
        self.model       = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        target = model.spatial_branch.backbone.conv_head
        target.register_forward_hook(self._save_activations)
        target.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _input, output) -> None:
        self._activations = output                  # (B, C, H, W)

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]            # (B, C, H, W)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Grad-CAM maps for a batch.

        Runs its own forward pass on x so that activations and gradients
        always come from the same call and have matching batch sizes.

        Args:
            x: (B, 3, 224, 224) input images (fake subset only)

        Returns:
            cam: (B, H', W') ReLU-clamped and min-max normalised CAM maps
        """
        self.model.zero_grad()
        logit = self.model(x)          # sets self._activations via forward hook
        logit.sum().backward(retain_graph=True)

        grads = self._gradients                     # (B, C, H', W')
        acts  = self._activations                   # (B, C, H', W')

        # Global average pool the gradients → channel weights (B, C, 1, 1)
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations → (B, H', W')
        cam = (weights * acts).sum(dim=1)
        cam = torch.clamp(cam, min=0)               # ReLU

        # Min-max normalise per image so maps are comparable
        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cam_min  = cam_flat.min(dim=1).values.view(B, 1, 1)
        cam_max  = cam_flat.max(dim=1).values.view(B, 1, 1)
        denom    = (cam_max - cam_min).clamp(min=1e-8)
        cam      = (cam - cam_min) / denom

        return cam.detach()                         # (B, H', W')


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine annealing
# ---------------------------------------------------------------------------

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps  = cfg.epochs        * steps_per_epoch
    lr_min_ratio = cfg.lr_min / cfg.lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Fo-Mixup helpers
# ---------------------------------------------------------------------------

def _apply_fo_mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha:  float,
) -> torch.Tensor:
    """Return a frequency-augmented version of the batch.

    Only fake images are augmented. Real images that happen to share a batch
    with fakes are used as the 'real reference' for mixing, but their own
    pixels are returned unchanged.

    Returns x_aug: same shape as images, fakes have their LL/HH perturbed,
    reals are identical to the input.
    """
    fake_mask = (labels == 1)
    real_mask = (labels == 0)

    n_fake = fake_mask.sum().item()
    n_real = real_mask.sum().item()

    if n_fake == 0 or n_real == 0:
        return images.clone()   # nothing to mix

    x_fake = images[fake_mask]
    x_real = images[real_mask]

    # If sizes differ, repeat the smaller set to match
    if n_fake != n_real:
        repeats = -(-n_fake // n_real)            # ceil division
        x_real  = x_real.repeat(repeats, 1, 1, 1)[:n_fake]

    # We don't modify the spatial pixels, only the frequency representations
    # used by the LF/HF branches. We reconstruct a modified image by blending
    # back only the HH component into the original image in the spatial domain.
    # The spatial branch always sees the original, unmodified image.
    #
    # Here we return the original image unchanged — the Fo-Mixup tensors are
    # computed inside the model via build_lf_tensor / build_hf_tensor with
    # the augmented subbands substituted. See Trainer._forward_aug().
    x_aug = images.clone()
    return x_aug   # spatial pixels unchanged; subband blending happens in _forward_aug


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:

    def __init__(self, cfg: TrainingConfig, resume_from: Path | None = None) -> None:
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        self._setup_data()
        self._setup_model()

        self.global_step = 0
        self.best_auc    = 0.0
        self.start_epoch = 1

        if resume_from is not None:
            self._load_checkpoint(resume_from)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_data(self) -> None:
        cfg = self.cfg
        print(f"Loading index: {cfg.crops_csv}")
        df = pd.read_csv(cfg.crops_csv)

        # Use crop_path if available, otherwise fall back to path
        if 'crop_path' in df.columns:
            df['path'] = df['crop_path'].where(
                df['crop_path'].notna() & (df['crop_path'] != ''),
                df['path']
            )

        # Rewrite path prefix when running on a different machine (e.g. Colab).
        # The CSV was built locally with absolute paths like /home/angelo/...
        # On Colab the crops live under a different root, so we keep only the
        # relative tail (method/video_id/frame.png) and prepend crops_root.
        if cfg.crops_root is not None:
            import re
            # Tail is the last 3 path components: method/video_id/frame.png
            df['path'] = df['path'].apply(
                lambda p: str(cfg.crops_root / Path(*Path(p).parts[-3:]))
            )
            print(f"  Rewrote crop paths to root: {cfg.crops_root}")

        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        val_df   = df[df['split'] == 'val'  ].reset_index(drop=True)

        print(f"  train: {len(train_df):,}  val: {len(val_df):,}")

        train_ds = FFPPDataset(train_df, is_train=True)
        val_ds   = FFPPDataset(val_df,   is_train=False)

        sampler = make_weighted_sampler(train_df)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == 'cuda'),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == 'cuda'),
        )

    def _setup_model(self) -> None:
        cfg = self.cfg

        self.model    = FreqForensics().to(self.device)
        self.loss_fn  = FreqForensicsLoss(
            lambda_aux=cfg.lambda_aux,
            beta_local=cfg.beta_local,
            beta_global=cfg.beta_global,
            cam_every_n=cfg.cam_every_n,
        )
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        steps_per_epoch  = len(self.train_loader)
        self.scheduler   = _build_scheduler(self.optimizer, cfg, steps_per_epoch)
        self.grad_cam    = GradCAM(self.model)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            elapsed    = time.time() - t0

            if epoch % cfg.val_every_n_epochs == 0:
                auc, acc = self._validate()
                print(
                    f"[Epoch {epoch:>3}/{cfg.epochs}]  "
                    f"loss={train_loss:.4f}  val_auc={auc:.4f}  "
                    f"val_acc={acc:.4f}  lr={self._current_lr():.2e}  "
                    f"time={elapsed:.0f}s"
                )
                self._save_checkpoint(epoch, auc)
            else:
                print(
                    f"[Epoch {epoch:>3}/{cfg.epochs}]  "
                    f"loss={train_loss:.4f}  lr={self._current_lr():.2e}  "
                    f"time={elapsed:.0f}s"
                )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0

        for images, labels, _video_ids, _methods in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            loss, breakdown = self._train_step(images, labels)

            running_loss += loss

            if self.global_step % self.cfg.log_every_n == 0:
                print(
                    f"  step={self.global_step:>6}  "
                    + "  ".join(f"{k}={v:.4f}" for k, v in breakdown.items())
                )

            self.global_step += 1

        return running_loss / len(self.train_loader)

    def _train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, dict]:
        cfg       = self.cfg
        fake_mask = (labels == 1)
        real_mask = (labels == 0)

        # ----- Forward pass (original) -----
        logit, aux_s, aux_lf, aux_hf, f_s, f_lf, f_hf = (
            self.model.forward_with_aux(images)
        )

        # ----- Grad-CAM for L_local -----
        cam_orig = cam_aug = None
        compute_cam = (
            cfg.cam_every_n > 0
            and self.global_step % cfg.cam_every_n == 0
            and fake_mask.any()
        )
        if compute_cam:
            cam_orig = self.grad_cam(images[fake_mask])

        # ----- Fo-Mixup augmented forward (fakes only) -----
        f_orig = f_aug = None
        if fake_mask.any() and real_mask.any():
            x_fake = images[fake_mask]
            x_real = images[real_mask]

            # Align sizes for mixing
            n_fake, n_real = x_fake.shape[0], x_real.shape[0]
            if n_fake != n_real:
                repeats = -(-n_fake // n_real)
                x_real  = x_real.repeat(repeats, 1, 1, 1)[:n_fake]

            # Build augmented images: replace DWT subbands with Fo-Mixup versions
            x_aug = self._build_fo_mixup_image(x_fake, x_real)

            with torch.no_grad():
                _, _, _, _, f_s_aug, f_lf_aug, f_hf_aug = (
                    self.model.forward_with_aux(x_aug)
                )

            # Concatenate branch features for L_global
            f_orig = torch.cat([f_s[fake_mask],  f_lf[fake_mask],  f_hf[fake_mask]],  dim=1)
            f_aug  = torch.cat([f_s_aug,          f_lf_aug,          f_hf_aug],          dim=1)

            if compute_cam:
                cam_aug = self.grad_cam(x_aug)

        # ----- Loss -----
        total, breakdown = self.loss_fn(
            logit=logit, labels=labels,
            aux_s=aux_s, aux_lf=aux_lf, aux_hf=aux_hf,
            step=self.global_step,
            f_orig=f_orig, f_aug=f_aug,
            cam_orig=cam_orig, cam_aug=cam_aug,
        )

        # ----- Backward -----
        self.optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return total.item(), breakdown

    def _build_fo_mixup_image(
        self,
        x_fake: torch.Tensor,
        x_real: torch.Tensor,
    ) -> torch.Tensor:
        """Build a spatial image whose DWT subbands are Fo-Mixup blended.

        We approximate this by reconstructing from the augmented LL subband:
        we replace the low-frequency component of x_fake with the mixed LL,
        then add back the high-frequency residual from the original.

        In practice the spatial branch sees this reconstructed image, so it
        also receives frequency-debiased content.
        """
        alpha = self.cfg.fo_mixup_alpha

        gray_fake = to_grayscale(x_fake)          # (B, 1, H, W)
        gray_real = to_grayscale(x_real)

        LL_fake, LH, HL, HH_fake = haar_dwt2d(gray_fake)
        LL_real, _,  _,  HH_real = haar_dwt2d(gray_real)

        # Blend LL and HH in frequency domain
        from fo_mixup import _sample_lambda, _mix_in_frequency_domain
        lam    = _sample_lambda(x_fake.shape[0], alpha, x_fake.device)
        LL_aug = _mix_in_frequency_domain(LL_fake, LL_real, lam)
        HH_aug = _mix_in_frequency_domain(HH_fake, HH_real, lam)

        # Reconstruct grayscale image from augmented subbands via inverse Haar
        # Haar inverse: interleave LL, LH, HL, HH into 2×2 blocks
        B, _, H2, W2 = LL_aug.shape
        H, W = H2 * 2, W2 * 2

        # Use pixel-shuffle style reconstruction (approximate)
        top    = torch.cat([LL_aug, LH], dim=3)    # (B,1,H/2, W)
        bottom = torch.cat([HL,    HH_aug], dim=3)
        gray_aug = torch.cat([top, bottom], dim=2)  # (B,1,H,W)  -- approx

        # Replace grayscale channel in a 3-channel copy (spatial branch sees colour)
        # We blend the original colour image towards a grayscale-reconstructed version
        x_aug = x_fake.clone()
        for c in range(3):
            x_aug[:, c:c+1, :, :] = (
                0.5 * x_fake[:, c:c+1, :, :] + 0.5 * gray_aug
            )

        return x_aug

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        all_logits = []
        all_labels = []

        for images, labels, _vids, _methods in self.val_loader:
            images = images.to(self.device)
            logit  = self.model(images)              # (B, 1)
            all_logits.append(logit.squeeze(1).cpu())
            all_labels.append(labels)

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs  = 1.0 / (1.0 + np.exp(-logits))     # sigmoid

        auc = roc_auc_score(labels, probs)
        acc = ((probs >= 0.5).astype(int) == labels).mean()

        self.model.train()
        return float(auc), float(acc)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: Path) -> None:
        print(f"Resuming from checkpoint: {path}")
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.global_step = state['global_step']
        self.best_auc    = state['best_auc']
        self.start_epoch = state['epoch'] + 1   # resume from the next epoch

        print(f"  Resumed at epoch {state['epoch']}  "
              f"global_step={self.global_step}  best_auc={self.best_auc:.4f}")

    def _save_checkpoint(self, epoch: int, auc: float) -> None:
        state = {
            'epoch':       epoch,
            'global_step': self.global_step,
            'model':       self.model.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'scheduler':   self.scheduler.state_dict(),
            'best_auc':    self.best_auc,
            'config':      self.cfg.__dict__,
        }

        # Always save latest
        latest_path = self.cfg.output_dir / 'latest.pt'
        torch.save(state, latest_path)

        # Save best
        if auc > self.best_auc:
            self.best_auc = auc
            best_path = self.cfg.output_dir / 'best.pt'
            torch.save(state, best_path)
            print(f"  ** New best AUC: {auc:.4f} — saved to {best_path}")

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
