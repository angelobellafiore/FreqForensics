import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
from transforms import train_transform, val_transform


class FFPPDataset(Dataset):
    """PyTorch Dataset for FaceForensics++ face crops.

    Reads a pre-built index CSV where each row is one extracted frame.
    The CSV must contain columns: path, label, video_id, method, split.

    The caller is responsible for filtering the DataFrame to the desired
    split before instantiating this class — the Dataset itself is split-agnostic.

    Args:
        df:       DataFrame filtered to the desired split (train / val / test)
        is_train: If True, applies train_transform (augmentations).
                  If False, applies val_transform (normalisation only).
    """

    def __init__(self, df: pd.DataFrame, is_train: bool = False) -> None:
        self.df        = df.reset_index(drop=True)
        self.transform = train_transform if is_train else val_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]

        image  = Image.open(row['path']).convert('RGB')
        tensor = self.transform(image)

        label    = int(row['label'])
        video_id = str(row['video_id'])
        method   = str(row['method'])

        return tensor, label, video_id, method


if __name__ == '__main__':
    import torch

    print("Running smoke test for dataset.py...")
    print("(Uses a synthetic in-memory DataFrame — no real data required)\n")

    import tempfile
    import numpy as np
    from torch.utils.data import DataLoader

    # Create a few temporary 224x224 PNG crops to simulate real data
    tmp_dir  = tempfile.mkdtemp()
    n_real   = 4
    n_fake   = 4
    records  = []

    for i in range(n_real + n_fake):
        path = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        # Random RGB image saved as PNG
        arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)

        label  = 0 if i < n_real else 1
        method = 'original' if label == 0 else 'Deepfakes'
        records.append({
            'path':     path,
            'label':    label,
            'video_id': f'{i:03d}',
            'method':   method,
            'split':    'train',
        })

    df = pd.DataFrame(records)

    # Test training dataset
    train_ds = FFPPDataset(df, is_train=True)
    assert len(train_ds) == n_real + n_fake, "Wrong dataset length"

    tensor, label, video_id, method = train_ds[0]
    assert tensor.shape == (3, 224, 224), f"Wrong tensor shape: {tensor.shape}"
    assert tensor.dtype == torch.float32,  f"Wrong dtype: {tensor.dtype}"
    assert label in (0, 1),                f"Unexpected label: {label}"
    print(f"  Single item: tensor{tuple(tensor.shape)}, label={label}, "
          f"video_id='{video_id}', method='{method}'  OK")

    # Test DataLoader batching
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch_tensor, batch_labels, batch_vids, batch_methods = next(iter(loader))
    assert batch_tensor.shape == (4, 3, 224, 224), f"Wrong batch shape: {batch_tensor.shape}"
    print(f"  DataLoader batch: tensors{tuple(batch_tensor.shape)}, "
          f"labels={batch_labels.tolist()}  OK")

    # Test val dataset (no augmentations)
    val_ds = FFPPDataset(df, is_train=False)
    tensor_val, _, _, _ = val_ds[0]
    assert tensor_val.shape == (3, 224, 224), f"Val tensor shape wrong: {tensor_val.shape}"
    print(f"  Val dataset (no augmentations): tensor{tuple(tensor_val.shape)}  OK")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)

    print("\nAll assertions passed.")
