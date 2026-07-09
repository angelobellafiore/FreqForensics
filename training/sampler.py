"""Weighted random sampler factory for FreqForensics.

Balances real vs. fake frames in each training batch by assigning each sample
a weight inversely proportional to its class frequency.
"""

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that yields ~50/50 real/fake batches.

    Args:
        df: DataFrame for the training split. Must have a 'label' column
            (0 = real, 1 = fake).

    Returns:
        WeightedRandomSampler with replacement=True.
    """
    labels = df['label'].values          # numpy array of 0s and 1s

    n_real = (labels == 0).sum()
    n_fake = (labels == 1).sum()

    print(f"  Sampler: n_real={n_real:,}  n_fake={n_fake:,}")

    # Weight per class: rarer class gets higher weight
    w_real = 1.0 / n_real
    w_fake = 1.0 / n_fake

    # One weight per sample
    sample_weights = torch.where(   # assign w_real to real samples, w_fake to fake samples
        torch.tensor(labels == 0),
        torch.tensor(w_real, dtype=torch.float64),
        torch.tensor(w_fake, dtype=torch.float64),
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


if __name__ == '__main__':
    print("Running smoke test for sampler.py...")

    # Simulate imbalanced dataset: 1000 real, 4000 fake
    df = pd.DataFrame({
        'label': [0] * 1_000 + [1] * 4_000,
        'split': ['train'] * 5_000,
    })

    sampler = make_weighted_sampler(df)

    # Draw one epoch worth of indices and check class balance
    from torch.utils.data import DataLoader, TensorDataset
    labels_tensor = torch.tensor(df['label'].values)
    dataset = TensorDataset(labels_tensor)
    loader  = DataLoader(dataset, batch_size=32, sampler=sampler)

    n_real = n_fake = 0
    for (batch_labels,) in loader:
        n_real += (batch_labels == 0).sum().item()
        n_fake += (batch_labels == 1).sum().item()

    total = n_real + n_fake
    print(f"  Drew {total:,} samples: real={n_real:,} ({n_real/total:.1%})  "
          f"fake={n_fake:,} ({n_fake/total:.1%})")
    assert 0.45 < n_real / total < 0.55, "Class balance is off, expected ~50/50"
    print("\nAll assertions passed.")
