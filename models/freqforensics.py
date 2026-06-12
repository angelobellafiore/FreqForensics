import torch
import torch.nn as nn

from freq_transforms import build_lf_tensor, build_hf_tensor
from lf_encoder import LowFreqEncoder
from hf_encoder import HighFreqEncoder
from spatial_branch import SpatialBranch
from fusion import CrossBranchAttentionFusion
from classifier_head import ClassifierHead, AuxiliaryHead


class FreqForensics(nn.Module):
    """Full triple-branch deepfake detector.

    Architecture:
      Branch 1 — Spatial:       EfficientNet-B4  -> (B, 1792)
      Branch 2 — Low-Frequency: LowFreqEncoder   -> (B, 256)
      Branch 3 — High-Frequency: HighFreqEncoder -> (B, 256)
      Fusion:   CrossBranchAttentionFusion        -> (B, 2304)
      Head:     ClassifierHead                    -> (B, 1)  logit

    Use forward() at inference time.
    Use forward_with_aux() during training to get auxiliary logits and
    branch feature vectors needed for the consistency losses.
    """

    def __init__(self) -> None:
        super().__init__()

        self.spatial_branch = SpatialBranch()
        self.lf_encoder     = LowFreqEncoder()
        self.hf_encoder     = HighFreqEncoder()
        self.fusion         = CrossBranchAttentionFusion()
        self.head           = ClassifierHead()

        # Auxiliary heads — training only, prevent branch collapse
        self.aux_s  = AuxiliaryHead(in_dim=1792)
        self.aux_lf = AuxiliaryHead(in_dim=256)
        self.aux_hf = AuxiliaryHead(in_dim=256)

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run all three branches and return raw feature vectors."""
        f_s  = self.spatial_branch(x)
        f_lf = self.lf_encoder(build_lf_tensor(x))
        f_hf = self.hf_encoder(build_hf_tensor(x))
        return f_s, f_lf, f_hf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward pass.

        Args:
            x: (B, 3, 224, 224) ImageNet-normalised RGB tensor
        Returns:
            (B, 1) logit — apply sigmoid for probability
        """
        f_s, f_lf, f_hf = self._encode(x)
        fused = self.fusion(f_s, f_lf, f_hf)
        return self.head(fused)

    def forward_with_aux(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass — returns logits and branch features.

        Args:
            x: (B, 3, 224, 224) ImageNet-normalised RGB tensor
        Returns:
            logit:   (B, 1)   primary fused logit
            aux_s:   (B, 1)   spatial branch auxiliary logit
            aux_lf:  (B, 1)   LF branch auxiliary logit
            aux_hf:  (B, 1)   HF branch auxiliary logit
            f_s:     (B, 1792) spatial feature vector
            f_lf:    (B, 256)  LF feature vector
            f_hf:    (B, 256)  HF feature vector
        """
        f_s, f_lf, f_hf = self._encode(x)
        fused = self.fusion(f_s, f_lf, f_hf)
        logit = self.head(fused)

        aux_s  = self.aux_s(f_s)
        aux_lf = self.aux_lf(f_lf)
        aux_hf = self.aux_hf(f_hf)

        return logit, aux_s, aux_lf, aux_hf, f_s, f_lf, f_hf


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    print("Running smoke test for freqforensics.py...")
    print("(Loading EfficientNet-B4 pretrained weights — may take a moment...)\n")

    model = FreqForensics()
    model.eval()

    B = 2
    x = torch.rand(B, 3, 224, 224)

    # Inference forward pass
    with torch.no_grad():
        logit = model(x)
    assert logit.shape == (B, 1), f"Expected ({B}, 1), got {logit.shape}"
    assert not torch.isnan(logit).any(), "logit contains NaN"
    print(f"  forward():          {tuple(x.shape)} -> {tuple(logit.shape)}  OK")

    # Training forward pass
    with torch.no_grad():
        logit, aux_s, aux_lf, aux_hf, f_s, f_lf, f_hf = model.forward_with_aux(x)
    assert logit.shape  == (B, 1),    f"logit:  {logit.shape}"
    assert aux_s.shape  == (B, 1),    f"aux_s:  {aux_s.shape}"
    assert aux_lf.shape == (B, 1),    f"aux_lf: {aux_lf.shape}"
    assert aux_hf.shape == (B, 1),    f"aux_hf: {aux_hf.shape}"
    assert f_s.shape    == (B, 1792), f"f_s:    {f_s.shape}"
    assert f_lf.shape   == (B, 256),  f"f_lf:   {f_lf.shape}"
    assert f_hf.shape   == (B, 256),  f"f_hf:   {f_hf.shape}"
    print(f"  forward_with_aux(): logit{tuple(logit.shape)}, "
          f"aux_s{tuple(aux_s.shape)}, aux_lf{tuple(aux_lf.shape)}, "
          f"aux_hf{tuple(aux_hf.shape)}  OK")
    print(f"                      f_s{tuple(f_s.shape)}, "
          f"f_lf{tuple(f_lf.shape)}, f_hf{tuple(f_hf.shape)}  OK")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    print("\nAll assertions passed.")
