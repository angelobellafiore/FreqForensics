import timm
import torch
import torch.nn as nn


class SpatialBranch(nn.Module):
    """EfficientNet-B4 spatial branch with partial fine-tuning.

    Input:  (B, 3, 224, 224) normalised RGB tensor
    Output: (B, 1792) spatial feature vector

    Blocks 0-5 are frozen, they extract low-level features (edges, textures,
    shapes) that transfer directly from ImageNet. Block 6, conv_head, and all
    BatchNorm layers are unfrozen for minor adaptation to face manipulation signals.

    This reduces trainable parameters from ~19M (full backbone) to ~5M,
    which is important given FF++'s limited training set size (720 source videos).
    """

    def __init__(self) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=True,
            num_classes=0,        # remove classification head
            global_pool='avg',    # global average pooling -> (B, 1792)
        )

        self._apply_freeze_strategy()

    def _apply_freeze_strategy(self) -> None:
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze block 6 (last MBConv block)
        for param in self.backbone.blocks[6].parameters():
            param.requires_grad = True

        # Unfreeze conv_head (1x1 conv after the blocks)
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True

        # Unfreeze all BatchNorm layers, their statistics must adapt
        # to the face crop distribution
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) ImageNet-normalised RGB tensor
        Returns:
            (B, 1792) spatial feature vector
        """
        return self.backbone(x)


if __name__ == '__main__':
    print("Running smoke test for spatial_branch.py...")

    model = SpatialBranch()
    model.eval()

    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 1792), f"Expected (2, 1792), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    print(f"  SpatialBranch: {tuple(x.shape)} -> {tuple(out.shape)}  OK")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}  (expected ~5M)")

    frozen_blocks = [str(i) for i in range(6)
                     if not any(p.requires_grad
                                for p in model.backbone.blocks[i].parameters())]
    print(f"  Frozen blocks: {frozen_blocks}  (expected ['0','1','2','3','4','5'])")

    print("\nAll assertions passed.")
