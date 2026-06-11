import torch
import torch.nn as nn


class LowFreqEncoder(nn.Module):
    """Encode the two-channel LF tensor (LL + LL2) into a 256-d feature vector.

    Input:  (B, 2, 112, 112)  — from build_lf_tensor()
    Output: (B, 256)

    Three strided conv blocks progressively halve the spatial resolution while
    doubling channels, then global average pooling collapses the spatial dims,
    and a linear projection maps to the shared 256-d feature space.

    Kept intentionally small (~300K params) so the encoder specialises in
    low-frequency structure anomalies rather than redundantly re-learning
    the spatial features already covered by the EfficientNet branch.
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: (B, 2, 112, 112) -> (B, 32, 56, 56)
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: (B, 32, 56, 56) -> (B, 64, 28, 28)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: (B, 64, 28, 28) -> (B, 128, 14, 14)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Global average pooling: (B, 128, 14, 14) -> (B, 128, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

        # Project from 128-d pooled representation to shared 256-d feature space
        self.projection = nn.Sequential(
            nn.Flatten(),               # (B, 128, 1, 1) -> (B, 128)
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 112, 112) normalised LF tensor from build_lf_tensor()
        Returns:
            (B, 256) LF feature vector
        """
        x = self.encoder(x)
        return self.projection(x)


if __name__ == '__main__':
    print("Running smoke test for lf_encoder.py...")

    model = LowFreqEncoder()
    model.eval()

    x = torch.rand(4, 2, 112, 112)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    print(f"  LowFreqEncoder: {tuple(x.shape)} -> {tuple(out.shape)}  OK")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  (expected ~300K)")

    print("\nAll assertions passed.")
