import torch
import torch.nn.functional as F


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, H, W) RGB tensor to (B, 1, H, W) grayscale using ITU-R BT.601 luminance weights.

    Manipulation artifacts are primarily in luminance — GAN generators operate
    mainly on grayscale structure, with colour as a secondary learned mapping.
    """
    weights = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)   # convert an RGB image to grayscale using luminance weights


def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-level 2D Haar DWT on a (B, 1, H, W) grayscale tensor.

    Returns four (B, 1, H/2, W/2) subbands: LL, LH, HL, HH.

    The Haar filters are the simplest possible wavelets:
      low-pass:  [1,  1] / sqrt(2)  — averages adjacent pixels
      high-pass: [1, -1] / sqrt(2)  — differences adjacent pixels

    Applying both filters along rows then columns yields four combinations:
      LL = low  x low  — coarse approximation (smooth structure)
      LH = low  x high — horizontal edges
      HL = high x low  — vertical edges
      HH = high x high — diagonal edges, fine textures, noise-like patterns
    """
    # 1D Haar filters as (out_channels, in_channels, kernel_size) conv kernels
    # We apply them as depthwise convolutions (groups=channels) with stride 2
    lo = x.new_tensor([1.0,  1.0]) / (2 ** 0.5)  # low-pass
    hi = x.new_tensor([1.0, -1.0]) / (2 ** 0.5)  # high-pass

    # Shape for a 1D horizontal convolution: (out_ch, in_ch, 1, kernel_w)
    lo_row = lo.view(1, 1, 1, 2)
    hi_row = hi.view(1, 1, 1, 2)

    # Apply filters along columns (horizontal direction) with stride 2
    # Result shape: (B, 1, H, W/2)
    L = F.conv2d(x,  lo_row, stride=(1, 2), padding=(0, 0))
    H = F.conv2d(x,  hi_row, stride=(1, 2), padding=(0, 0))

    # Shape for a 1D vertical convolution: (out_ch, in_ch, kernel_h, 1)
    lo_col = lo.view(1, 1, 2, 1)
    hi_col = hi.view(1, 1, 2, 1)

    # Apply filters along rows (vertical direction) with stride 2
    # Result shape: (B, 1, H/2, W/2)
    LL = F.conv2d(L, lo_col, stride=(2, 1), padding=(0, 0))     # Smooth subsampling of the original image. It's a miniature, blurred version of the input that retains almost all of the signal's energy.
    LH = F.conv2d(L, hi_col, stride=(2, 1), padding=(0, 0))     # Captures horizontal edges and textures (preserves the structure) — where pixel values change significantly across columns but are relatively constant across rows. (Highlights horizontal lines. This is achieved by applying a low-pass filter to the rows (preserving the structure) and a high-pass filter to the columns (detecting vertical variations).)
    HL = F.conv2d(H, lo_col, stride=(2, 1), padding=(0, 0))     # Captures vertical edges and textures (preserves the structure) — where pixel values change significantly across rows but are relatively constant across columns. (Highlights vertical lines. This is achieved by applying a high-pass filter to the rows (detecting horizontal variations) and a low-pass filter to the columns (preserving the structure).)
    HH = F.conv2d(H, hi_col, stride=(2, 1), padding=(0, 0))     # Captures diagonal edges, fine textures, and noise-like patterns.

    return LL, LH, HL, HH


def _log_minmax_normalise(x: torch.Tensor) -> torch.Tensor:
    """Apply log(|x| + 1) compression then per-image min-max normalisation to [0, 1].

    Log compression reduces dynamic range into the [0, 1] range, so CNN encoders see numerically
    well-conditioned inputs. The +1 prevents log(0) = -inf.
    Per-image (not per-batch) normalisation preserves relative differences
    between images while making the scale consistent.
    """
    x = torch.log(x.abs() + 1.0)    # Reduces the dynamic range. If the input has huge peaks of value (e.g., pixels at 10,000 and others at 1), the logarithm squashes the higher values, making the contrasts smoother. This helps the CNN converge better without being destabilized by out-of-scale values.
    # Normalise each image independently — flatten spatial dims to find min/max
    b = x.shape[0]
    x_flat = x.view(b, -1)      # All dimensions after the first (channels, height, width) are merged into a single one-dimensional vector for each element of the batch.
    x_min = x_flat.min(dim=1).values.view(b, 1, 1, 1)
    x_max = x_flat.max(dim=1).values.view(b, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + 1e-8) # Linear normalization formula. This operation rigidly maps all original values ​​into the exact interval [0, 1]. Add small epsilon to avoid division by zero if max == min


def build_lf_tensor(x: torch.Tensor) -> torch.Tensor:
    """Build the two-channel low-frequency tensor from an RGB input.

      1. Convert to grayscale
      2. Level-1 DWT → keep LL subband  (B, 1, H/2, W/2)
      3. Level-2 DWT on LL → keep LL2   (B, 1, H/4, W/4)        # second scale captures coarser structure/artifacts
      4. Upsample LL2 back to H/2 x W/2
      5. Log + min-max normalise each channel independently
      6. Concatenate LL and LL2 → (B, 2, H/2, W/2)

    The two-scale representation gives the LF encoder context at both the
    full half-resolution and a coarser quarter-resolution — useful for
    diffusion-model fakes whose artifacts span different spatial scales.

    Modern generative models leave specific fingerprints (artifacts) 
    in the low-frequency components of images. These anomalies are often
    invisible to the naked eye but appear as unnatural geometric regularities
    or anomalous statistical distributions.
    """
    gray = to_grayscale(x)                          # (B, 1, H, W)

    LL, _, _, _ = haar_dwt2d(gray)                  # (B, 1, H/2, W/2)
    LL2, _, _, _ = haar_dwt2d(LL)                   # (B, 1, H/4, W/4)

    LL2_up = F.interpolate(LL2, size=LL.shape[2:], mode='bilinear', align_corners=False)    # Upsample LL2 back to H/2 x W/2 using bilinear interpolation. This allows the LF encoder to have access to both the finer details captured in LL and the coarser, more global structure captured in LL2, which can be crucial for detecting artifacts that manifest at different scales. (Since LL2 is smaller in size than LL, they cannot be directly concatenated. By upsampling LL2 to match the spatial dimensions of LL, we can concatenate them along the channel dimension, allowing the model to learn from both levels of frequency information simultaneously.)

    LL_norm  = _log_minmax_normalise(LL)
    LL2_norm = _log_minmax_normalise(LL2_up)

    return torch.cat([LL_norm, LL2_norm], dim=1)    # (B, 2, H/2, W/2). The two normalized channels are merged along the channel dimension (dim=1).


def build_hf_tensor(x: torch.Tensor) -> torch.Tensor:
    """Build the three-channel high-frequency tensor from an RGB input.

      1. Convert to grayscale
      2. Level-1 DWT → keep LH, HL, HH subbands  each (B, 1, H/2, W/2)
      3. Log + min-max normalise each channel independently
      4. Concatenate → (B, 3, H/2, W/2)

    No second decomposition level here — level-1 detail subbands already
    capture fine-grained artifacts at the relevant scale. Deeper levels
    average out localised boundary artifacts.
    """
    gray = to_grayscale(x)                          # (B, 1, H, W)

    _, LH, HL, HH = haar_dwt2d(gray)               # each (B, 1, H/2, W/2)

    # Micro-Artifact Enhancement: Logarithmic compression squashes extremely
    # sharp edges (e.g., sharp transitions from white to black) and numerically
    # amplifies very faint background noise patterns, making them visible
    # and processable by subsequent convolutional layers of the CNN.
    # For high frequencies, it's not necessary to go down to Level 2 (H/4 X W/4)).
    # Diffusion models and GAN generators leave sampling anomalies
    # (e.g., checkerboard artifacts or unnatural spectral patterns) that are positioned
    # right at the level of individual nearby pixels. Going further down would mean
    # applying a low-pass filter that would blur and average out these micro-artifacts,
    # erasing the fingerprint of the Deepfake you're trying to detect.
    LH_norm = _log_minmax_normalise(LH)
    HL_norm = _log_minmax_normalise(HL)
    HH_norm = _log_minmax_normalise(HH)

    return torch.cat([LH_norm, HL_norm, HH_norm], dim=1)   # (B, 3, H/2, W/2)


if __name__ == '__main__':
    # Smoke test — run with: python models/freq_transforms.py
    print("Running smoke test for freq_transforms.py...")

    B, H, W = 4, 224, 224
    x = torch.rand(B, 3, H, W)   # fake batch of 4 RGB images at 224x224

    # Test grayscale conversion
    gray = to_grayscale(x)
    assert gray.shape == (B, 1, H, W), f"Expected ({B},1,{H},{W}), got {gray.shape}"
    print(f"  to_grayscale:   {tuple(x.shape)} -> {tuple(gray.shape)}  OK")

    # Test single-level DWT
    LL, LH, HL, HH = haar_dwt2d(gray)
    for name, band in [('LL', LL), ('LH', LH), ('HL', HL), ('HH', HH)]:
        assert band.shape == (B, 1, H//2, W//2), f"{name}: {band.shape}"
    print(f"  haar_dwt2d:     {tuple(gray.shape)} -> 4x {tuple(LL.shape)}  OK")

    # Test low-frequency tensor
    lf = build_lf_tensor(x)
    assert lf.shape == (B, 2, H//2, W//2), f"Expected ({B},2,{H//2},{W//2}), got {lf.shape}"
    assert lf.min() >= 0.0 and lf.max() <= 1.0, "LF tensor values out of [0,1] range"
    print(f"  build_lf_tensor: {tuple(x.shape)} -> {tuple(lf.shape)}  OK")

    # Test high-frequency tensor
    hf = build_hf_tensor(x)
    assert hf.shape == (B, 3, H//2, W//2), f"Expected ({B},3,{H//2},{W//2}), got {hf.shape}"
    assert hf.min() >= 0.0 and hf.max() <= 1.0, "HF tensor values out of [0,1] range"
    print(f"  build_hf_tensor: {tuple(x.shape)} -> {tuple(hf.shape)}  OK")

    print("\nAll assertions passed.")
