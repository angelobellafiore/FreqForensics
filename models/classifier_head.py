import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """MLP classifier over the 2304-d fused feature vector.

    Input:  (B, 2304) , from CrossBranchAttentionFusion
    Output: (B, 1)    , single logit (sigmoid applied at loss/inference time)

    BatchNorm1d normalises the combined spatial+frequency distribution, which
    can have very different scales across the two components. Dropout(0.5)
    provides strong regularisation at the fusion point where overfitting to
    method-specific feature combinations is most likely.
    """

    def __init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2304, 256),
            nn.BatchNorm1d(256),        # 1d: input is a flat vector, not a spatial map
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),          # strong regularisation at the fusion point
            nn.Linear(256, 1),          # sigmoid applied at loss/inference time
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2304) fused feature vector
        Returns:
            (B, 1) logit, no sigmoid applied here
        """
        return self.mlp(x)


class AuxiliaryHead(nn.Module):
    """Single-layer classifier attached directly to one branch's feature vector.

    Used during training only to prevent branch collapse, forces each branch
    to independently carry the classification signal. Detached at inference time.

    Input:  (B, in_dim) , raw branch feature vector before fusion
    Output: (B, 1)      , logit
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


if __name__ == '__main__':
    print("Running smoke test for classifier_head.py...")

    B = 4

    # Main classifier head
    head = ClassifierHead()
    head.eval()
    x = torch.rand(B, 2304)
    with torch.no_grad():
        out = head(x)
    assert out.shape == (B, 1), f"Expected ({B}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    print(f"  ClassifierHead: {tuple(x.shape)} -> {tuple(out.shape)}  OK")

    # Auxiliary heads for each branch
    for name, in_dim in [('spatial', 1792), ('lf', 256), ('hf', 256)]:
        aux = AuxiliaryHead(in_dim)
        aux.eval()
        x_aux = torch.rand(B, in_dim)
        with torch.no_grad():
            out_aux = aux(x_aux)
        assert out_aux.shape == (B, 1), f"AuxHead {name}: Expected ({B}, 1), got {out_aux.shape}"
        print(f"  AuxiliaryHead ({name}): {tuple(x_aux.shape)} -> {tuple(out_aux.shape)}  OK")

    print("\nAll assertions passed.")
