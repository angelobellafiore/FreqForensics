import torch
import torch.nn.functional as F


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """Convert an RGB tensor (B, 3, H, W) to grayscale (B, 1, H, W).

    Uses ITU-R BT.601 luminance weights. Forgery artifacts are primarily
    in luminance, so working on a single channel is sufficient for the DWT.
    """
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a single-level 2D Haar DWT to a grayscale tensor (B, 1, H, W).

    Returns four (B, 1, H/2, W/2) subbands:
      LL: low  x low,  coarse approximation (blurred downsampled image)
      LH: low  x high, horizontal edges
      HL: high x low,  vertical edges
      HH: high x high, diagonal edges and fine textures

    Implemented as two sequential 1D convolutions with stride 2,
    applied first along columns then along rows.
    """
    lo = x.new_tensor([1.0,  1.0]) / (2 ** 0.5)   # low-pass filter
    hi = x.new_tensor([1.0, -1.0]) / (2 ** 0.5)   # high-pass filter

    # Apply along columns (horizontal) with stride 2: (B,1,H,W) -> (B,1,H,W/2)
    L = F.conv2d(x, lo.view(1, 1, 1, 2), stride=(1, 2))
    H = F.conv2d(x, hi.view(1, 1, 1, 2), stride=(1, 2))

    # Apply along rows (vertical) with stride 2: (B,1,H,W/2) -> (B,1,H/2,W/2)
    LL = F.conv2d(L, lo.view(1, 1, 2, 1), stride=(2, 1))   # low  x low
    LH = F.conv2d(L, hi.view(1, 1, 2, 1), stride=(2, 1))   # low  x high
    HL = F.conv2d(H, lo.view(1, 1, 2, 1), stride=(2, 1))   # high x low
    HH = F.conv2d(H, hi.view(1, 1, 2, 1), stride=(2, 1))   # high x high

    return LL, LH, HL, HH


def _log_minmax_normalise(x: torch.Tensor) -> torch.Tensor:
    """Apply log(|x|+1) compression then rescale each image to [0, 1].

    Log compression reduces the dynamic range so CNN encoders receive
    numerically stable inputs. Per-image (not per-batch) normalisation
    preserves relative differences between images.
    """
    x = torch.log(x.abs() + 1.0)          # compress dynamic range
    b = x.shape[0]
    x_flat = x.view(b, -1)                # flatten spatial dims to find per-image min/max
    x_min = x_flat.min(dim=1).values.view(b, 1, 1, 1)
    x_max = x_flat.max(dim=1).values.view(b, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + 1e-8)   # +1e-8 avoids division by zero


def build_lf_tensor(x: torch.Tensor) -> torch.Tensor:
    """Build the two-channel low-frequency input for the LF encoder.

    Steps:
      1. Convert to grayscale
      2. Level-1 DWT: extract LL subband (B, 1, H/2, W/2)
      3. Level-2 DWT on LL: extract LL2 (B, 1, H/4, W/4), capturing coarser structure
      4. Upsample LL2 back to H/2 x W/2 so both channels have the same spatial size
      5. Log + min-max normalise each channel independently
      6. Concatenate: (B, 2, H/2, W/2)

    Two scales are used because diffusion-model fakes can leave artifacts
    at different spatial frequencies, and the two-channel input lets the
    LF encoder see both the full and coarser resolution simultaneously.
    """
    gray = to_grayscale(x)                          # (B, 1, H, W)

    LL,  _, _, _ = haar_dwt2d(gray)                 # (B, 1, H/2, W/2)
    LL2, _, _, _ = haar_dwt2d(LL)                   # (B, 1, H/4, W/4)

    # Upsample LL2 to match LL's spatial size before concatenating
    LL2_up = F.interpolate(LL2, size=LL.shape[2:], mode='bilinear', align_corners=False)

    LL_norm  = _log_minmax_normalise(LL)
    LL2_norm = _log_minmax_normalise(LL2_up)

    return torch.cat([LL_norm, LL2_norm], dim=1)    # (B, 2, H/2, W/2)


def build_hf_tensor(x: torch.Tensor) -> torch.Tensor:
    """Build the three-channel high-frequency input for the HF encoder.

    Steps:
      1. Convert to grayscale
      2. Level-1 DWT: extract LH, HL, HH subbands, each (B, 1, H/2, W/2)
      3. Log + min-max normalise each channel independently
      4. Concatenate: (B, 3, H/2, W/2)

    Only level-1 is used because GAN upsampling artifacts (checkerboard
    patterns, texture residuals) are localised at pixel level. Going deeper
    would apply additional averaging that blurs and erases these fingerprints.
    """
    gray = to_grayscale(x)                          # (B, 1, H, W)

    _, LH, HL, HH = haar_dwt2d(gray)               # each (B, 1, H/2, W/2)

    # Log-normalise each subband before stacking
    LH_norm = _log_minmax_normalise(LH)
    HL_norm = _log_minmax_normalise(HL)
    HH_norm = _log_minmax_normalise(HH)

    return torch.cat([LH_norm, HL_norm, HH_norm], dim=1)   # (B, 3, H/2, W/2)


if __name__ == '__main__':
    # Smoke test, run with: python models/freq_transforms.py
    print("Running smoke test for freq_transforms.py...")

    B, H, W = 4, 224, 224
    x = torch.rand(B, 3, H, W)

    gray = to_grayscale(x)
    assert gray.shape == (B, 1, H, W), f"Expected ({B},1,{H},{W}), got {gray.shape}"
    print(f"  to_grayscale:   {tuple(x.shape)} -> {tuple(gray.shape)}  OK")

    LL, LH, HL, HH = haar_dwt2d(gray)
    for name, band in [('LL', LL), ('LH', LH), ('HL', HL), ('HH', HH)]:
        assert band.shape == (B, 1, H//2, W//2), f"{name}: {band.shape}"
    print(f"  haar_dwt2d:     {tuple(gray.shape)} -> 4x {tuple(LL.shape)}  OK")

    lf = build_lf_tensor(x)
    assert lf.shape == (B, 2, H//2, W//2), f"Expected ({B},2,{H//2},{W//2}), got {lf.shape}"
    assert lf.min() >= 0.0 and lf.max() <= 1.0, "LF tensor values out of [0,1] range"
    print(f"  build_lf_tensor: {tuple(x.shape)} -> {tuple(lf.shape)}  OK")

    hf = build_hf_tensor(x)
    assert hf.shape == (B, 3, H//2, W//2), f"Expected ({B},3,{H//2},{W//2}), got {hf.shape}"
    assert hf.min() >= 0.0 and hf.max() <= 1.0, "HF tensor values out of [0,1] range"
    print(f"  build_hf_tensor: {tuple(x.shape)} -> {tuple(hf.shape)}  OK")

    print("\nAll assertions passed.")
