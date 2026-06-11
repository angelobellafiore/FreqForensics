import torch
import torch.nn as nn


class HighFreqEncoder(nn.Module):
    """Encode the three-channel HF tensor (LH + HL + HH) into a 256-d feature vector.

    Input:  (B, 3, 112, 112)  — from build_hf_tensor()
    Output: (B, 256)

    Structurally identical to LowFreqEncoder but with 3 input channels instead
    of 2, matching the three detail subbands (horizontal, vertical, diagonal).
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: (B, 3, 112, 112) -> (B, 32, 56, 56)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
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

        self.projection = nn.Sequential(
            nn.Flatten(),               # (B, 128, 1, 1) -> (B, 128)
            nn.Linear(128, 256),        # projected in the same shared 256-d latent space as the LF branch
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 112, 112) normalised HF tensor from build_hf_tensor()
        Returns:
            (B, 256) HF feature vector
        """
        x = self.encoder(x)
        return self.projection(x)


if __name__ == '__main__':
    print("Running smoke test for hf_encoder.py...")

    model = HighFreqEncoder()
    model.eval()

    x = torch.rand(4, 3, 112, 112)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    print(f"  HighFreqEncoder: {tuple(x.shape)} -> {tuple(out.shape)}  OK")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  (expected ~300K)")

    print("\nAll assertions passed.")
