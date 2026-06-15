"""FreqForensics composite training loss.

L_total = L_BCE + lambda * L_aux + beta1 * L_local + beta2 * L_global

L_BCE   — primary binary cross-entropy on the fused logit
L_aux   — sum of BCE on the three auxiliary branch logits (prevents branch collapse)
L_local — CAM alignment: L2 distance between Grad-CAM maps of original and
           augmented fake (forces spatially consistent forgery localisation)
L_global — vMF cosine consistency: 1 - cosine_similarity between L2-normalised
            feature vectors of original and augmented fake
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqForensicsLoss(nn.Module):
    """Composite loss for the FreqForensics training loop. Default values from the FreqDebias paper.

    Args:
        lambda_aux:  weight for auxiliary branch BCE losses (default 0.1)
        beta_local:  weight for CAM alignment loss            (default 0.5)
        beta_global: weight for vMF cosine consistency loss   (default 0.5)
        cam_every_n: compute L_local every N steps; 0 to disable (default 10)
    """

    def __init__(
        self,
        lambda_aux:  float = 0.1,
        beta_local:  float = 0.5,
        beta_global: float = 0.5,
        cam_every_n: int   = 10,
    ) -> None:
        super().__init__()
        self.lambda_aux  = lambda_aux
        self.beta_local  = beta_local
        self.beta_global = beta_global
        self.cam_every_n = cam_every_n

        self.bce = nn.BCEWithLogitsLoss()

    def l_aux(
        self,
        aux_s:  torch.Tensor,   # (B, 1) spatial auxiliary logit
        aux_lf: torch.Tensor,   # (B, 1) LF auxiliary logit
        aux_hf: torch.Tensor,   # (B, 1) HF auxiliary logit
        labels: torch.Tensor,   # (B,)   binary labels
    ) -> torch.Tensor:
        """Sum of BCE losses on the three branch auxiliary logits."""
        target = labels.float().unsqueeze(1)   # (B, 1)
        return (
            self.bce(aux_s,  target) +
            self.bce(aux_lf, target) +
            self.bce(aux_hf, target)
        )

    @staticmethod # no learnable parameters, so can be staticmethod
    def l_local(
        cam_orig: torch.Tensor,   # (B, H, W) Grad-CAM of original fakes
        cam_aug:  torch.Tensor,   # (B, H, W) Grad-CAM of augmented fakes
    ) -> torch.Tensor:
        """L2 distance between CAM maps — forces spatial cCalculating loss (like Cross-Entropy Loss) directly on logits rather than squashed probabilitiesonsistency under
        frequency perturbation."""
        return F.mse_loss(cam_orig, cam_aug)

    @staticmethod
    def l_global(
        f_orig: torch.Tensor,   # (B, D) feature vector of original fakes
        f_aug:  torch.Tensor,   # (B, D) feature vector of augmented fakes
    ) -> torch.Tensor:
        """vMF cosine consistency: 1 - cosine_similarity between L2-normalised
        feature vectors. Drives the model to produce similar representations
        for a fake and its frequency-perturbed version."""
        f_orig_norm = F.normalize(f_orig, dim=1)
        f_aug_norm  = F.normalize(f_aug,  dim=1)
        cos_sim = (f_orig_norm * f_aug_norm).sum(dim=1)   # (B,)
        return (1.0 - cos_sim).mean()

    def forward(
        self,
        logit:    torch.Tensor,         # (B, 1)  primary fused logit
        labels:   torch.Tensor,         # (B,)    binary labels
        aux_s:    torch.Tensor,         # (B, 1)  spatial auxiliary logit
        aux_lf:   torch.Tensor,         # (B, 1)  LF auxiliary logit
        aux_hf:   torch.Tensor,         # (B, 1)  HF auxiliary logit
        step:     int,                  # current training step (for cam_every_n)
        f_orig:   torch.Tensor | None = None,  # (B, D) features of original fakes
        f_aug:    torch.Tensor | None = None,  # (B, D) features of augmented fakes
        cam_orig: torch.Tensor | None = None,  # (B, H, W) CAM of original fakes
        cam_aug:  torch.Tensor | None = None,  # (B, H, W) CAM of augmented fakes
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total loss and return a breakdown dict for logging.

        Returns:
            total:    scalar loss tensor (differentiable)
            breakdown: {term: float} for logging — detached from graph
        """
        target = labels.float().unsqueeze(1)   # (B, 1)

        loss_bce = self.bce(logit, target)
        loss_aux = self.l_aux(aux_s, aux_lf, aux_hf, labels)

        loss_global = torch.tensor(0.0, device=logit.device)
        if f_orig is not None and f_aug is not None:
            loss_global = self.l_global(f_orig, f_aug)

        loss_local = torch.tensor(0.0, device=logit.device)
        compute_cam = (
            self.cam_every_n > 0
            and step % self.cam_every_n == 0
            and cam_orig is not None
            and cam_aug is not None
        )
        if compute_cam:
            loss_local = self.l_local(cam_orig, cam_aug)

        total = (
            loss_bce
            + self.lambda_aux  * loss_aux
            + self.beta_global * loss_global
            + self.beta_local  * loss_local
        )

        breakdown = {
            'loss_total':  total.item(),
            'loss_bce':    loss_bce.item(),
            'loss_aux':    loss_aux.item(),
            'loss_global': loss_global.item(),
            'loss_local':  loss_local.item(),
        }

        return total, breakdown


if __name__ == '__main__':
    print("Running smoke test for loss.py...")

    B, D = 8, 256
    loss_fn = FreqForensicsLoss()

    logit  = torch.randn(B, 1, requires_grad=True)  # we have to specify requires_grad=True, otherwise by default it is False and we need to know the gradients in order to do .backward()
    labels = torch.randint(0, 2, (B,))
    aux_s  = torch.randn(B, 1, requires_grad=True)
    aux_lf = torch.randn(B, 1, requires_grad=True)
    aux_hf = torch.randn(B, 1, requires_grad=True)
    f_orig = torch.randn(B, D)
    f_aug  = torch.randn(B, D)
    cam_o  = torch.rand(B, 14, 14)
    cam_a  = torch.rand(B, 14, 14)

    # Step where CAM is computed (step=0, cam_every_n=10 -> 0%10==0 -> True)
    total, breakdown = loss_fn(
        logit, labels, aux_s, aux_lf, aux_hf,
        step=0, f_orig=f_orig, f_aug=f_aug,
        cam_orig=cam_o, cam_aug=cam_a,
    )
    assert not torch.isnan(total), "Loss is NaN"
    assert breakdown['loss_local'] > 0, "L_local should be > 0 at step 0"
    print(f"  step=0  (CAM computed): {breakdown}")

    # Step where CAM is skipped
    total2, breakdown2 = loss_fn(
        logit, labels, aux_s, aux_lf, aux_hf,
        step=1, f_orig=f_orig, f_aug=f_aug,
        cam_orig=cam_o, cam_aug=cam_a,
    )
    assert breakdown2['loss_local'] == 0.0, "L_local should be 0 at step 1"
    print(f"  step=1  (CAM skipped):  {breakdown2}")

    # Backward pass
    total.backward()
    print(f"\n  Backward pass: OK")

    print("\nAll assertions passed.")
