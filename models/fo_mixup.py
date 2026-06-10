import torch
from freq_transforms import to_grayscale, haar_dwt2d


def _sample_lambda(batch_size: int, alpha: float, device: torch.device) -> torch.Tensor:
    """Sample per-image mixing coefficients from Beta(alpha, alpha).

    With alpha=1.0, Beta(1,1) = Uniform(0,1): every mixing ratio is equally
    likely. Higher alpha concentrates lambda around 0.5 (stronger mixing).

    Returns shape (batch_size, 1, 1, 1) so it broadcasts over (B, C, H, W).
    """
    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample((batch_size,))
    return lam.to(device).view(batch_size, 1, 1, 1)


def _mix_in_frequency_domain(
    subband_fake: torch.Tensor,
    subband_real: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """Blend two subbands in the frequency domain and reconstruct.

    Steps:
      1. FFT both subbands into complex frequency representations
      2. Linearly interpolate: F_mixed = lam * F_fake + (1 - lam) * F_real
      3. IFFT back to spatial domain
      4. Take the real part (imaginary residuals are numerical noise)

    Working in the frequency domain means the blending happens across all
    frequencies simultaneously. A spatial blend would just alpha-composite
    pixels, which doesn't disrupt the spectral fingerprint.
    """
    F_fake = torch.fft.rfft2(subband_fake)
    F_real = torch.fft.rfft2(subband_real)

    F_mixed = lam * F_fake + (1.0 - lam) * F_real

    return torch.fft.irfft2(F_mixed, s=subband_fake.shape[-2:]).real


def subband_fo_mixup_ll(
    x_fake: torch.Tensor,
    x_real: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply Fo-Mixup to the LL subband for the low-frequency branch.

    Called inside the training loop (not the DataLoader) because it needs
    both fake and real images from the same batch.

    Args:
        x_fake: (B, 3, H, W) batch of fake images
        x_real: (B, 3, H, W) batch of real images, same size as x_fake
        alpha:  Beta distribution concentration parameter (default 1.0 = Uniform)

    Returns:
        LL_aug: (B, 1, H/2, W/2) augmented LL subband.
                Pass this to the LowFreqEncoder in place of the original LL
                when computing consistency losses.
    """
    # independently converted to grayscale to isolate geometric and luminance structures.
    gray_fake = to_grayscale(x_fake)
    gray_real = to_grayscale(x_real)

    LL_fake, _, _, _ = haar_dwt2d(gray_fake)   # (B, 1, H/2, W/2)
    LL_real, _, _, _ = haar_dwt2d(gray_real)   # (B, 1, H/2, W/2)

    lam = _sample_lambda(x_fake.shape[0], alpha, x_fake.device)

    return _mix_in_frequency_domain(LL_fake, LL_real, lam)


def subband_fo_mixup_hh(
    x_fake: torch.Tensor,
    x_real: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply Fo-Mixup to the HH subband only for the high-frequency branch.

    LH and HL are intentionally NOT mixed — they carry horizontal and vertical
    edge information needed intact to detect blend boundary artifacts. Only HH
    (the subband most concentrated with GAN upsampling signatures: upsampling
    grids, texture over-smoothing, blend boundary residuals) is perturbed.

    Args:
        x_fake: (B, 3, H, W) batch of fake images
        x_real: (B, 3, H, W) batch of real images, same size as x_fake
        alpha:  Beta distribution concentration parameter (default 1.0 = Uniform)

    Returns:
        HH_aug: (B, 1, H/2, W/2) augmented HH subband.
                Reassemble the three-channel HF tensor by replacing the HH
                channel with this before passing to the HighFreqEncoder.
    """
    gray_fake = to_grayscale(x_fake)
    gray_real = to_grayscale(x_real)

    _, _, _, HH_fake = haar_dwt2d(gray_fake)   # (B, 1, H/2, W/2)
    _, _, _, HH_real = haar_dwt2d(gray_real)   # (B, 1, H/2, W/2)

    lam = _sample_lambda(x_fake.shape[0], alpha, x_fake.device)

    return _mix_in_frequency_domain(HH_fake, HH_real, lam)


if __name__ == '__main__':
    print("Running smoke test for fo_mixup.py...")

    B, H, W = 4, 224, 224
    x_fake = torch.rand(B, 3, H, W)
    x_real = torch.rand(B, 3, H, W)

    # Test LL mixup
    LL_aug = subband_fo_mixup_ll(x_fake, x_real, alpha=1.0)
    assert LL_aug.shape == (B, 1, H // 2, W // 2), f"LL_aug shape wrong: {LL_aug.shape}"
    assert not torch.isnan(LL_aug).any(), "LL_aug contains NaN"
    print(f"  subband_fo_mixup_ll: {tuple(x_fake.shape)} -> {tuple(LL_aug.shape)}  OK")

    # Test HH mixup
    HH_aug = subband_fo_mixup_hh(x_fake, x_real, alpha=1.0)
    assert HH_aug.shape == (B, 1, H // 2, W // 2), f"HH_aug shape wrong: {HH_aug.shape}"
    assert not torch.isnan(HH_aug).any(), "HH_aug contains NaN"
    print(f"  subband_fo_mixup_hh: {tuple(x_fake.shape)} -> {tuple(HH_aug.shape)}  OK")

    # Verify the augmented subband is neither identical to fake nor real
    # (with random inputs and Uniform lambda, the chance of exact equality is ~0)
    gray_fake = to_grayscale(x_fake)
    gray_real = to_grayscale(x_real)
    _, _, _, HH_fake = haar_dwt2d(gray_fake)
    _, _, _, HH_real = haar_dwt2d(gray_real)
    assert not torch.allclose(HH_aug, HH_fake), "HH_aug identical to HH_fake — mixing had no effect"
    assert not torch.allclose(HH_aug, HH_real), "HH_aug identical to HH_real — mixing had no effect"
    print("  Mixing verification: augmented != fake and augmented != real  OK")

    print("\nAll assertions passed.")
