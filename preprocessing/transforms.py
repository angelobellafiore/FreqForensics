from torchvision import transforms

# ImageNet statistics, required for the pretrained EfficientNet-B4 backbone.
# The backbone was trained with these exact values; its BatchNorm layers expect
# inputs centred at this mean with this standard deviation.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# Applied during training only.
# Each augmentation is chosen to increase robustness without destroying
# the frequency signals the two DWT branches need to detect.
#
# NOT included:
#   - Heavy Gaussian blur  -> destroys the HH subband signal
#   - JPEG re-compression  -> c23 already has JPEG-like artifacts; adding more
#                            washes out frequency cues
#   - CutMix / CutOut      -> removes spatial regions that may contain the blend
#                            boundary
#   - Frequency masking    -> the only frequency augmentation is Fo-Mixup,
#                            applied inside the training loop, not here
train_transform = transforms.Compose([
    # Faces are approximately symmetric, flipping doubles effective training
    # data without altering manipulation signals
    transforms.RandomHorizontalFlip(p=0.5),

    # Increases robustness to lighting variation across FF++ source videos.
    # Small values: we don't want colour shifts large enough to alter the
    # spectral fingerprint we're trying to detect.
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.05,
    ),

    # Increases robustness to slight landmark alignment variation.
    # Small values: large rotations or translations would shift the blend
    # boundary out of the crop region.
    transforms.RandomAffine(
        degrees=5,
        translate=(0.02, 0.02),
        scale=(0.98, 1.02),
    ),

    transforms.ToTensor(),                          # PIL [0,255] -> tensor [0,1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# Applied at validation and test time.
# No augmentations, deterministic and reproducible.
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


if __name__ == '__main__':
    import torch
    from PIL import Image
    import numpy as np

    print("Running smoke test for transforms.py...")

    # Simulate a 224x224 RGB face crop (output of face_detector)
    dummy_crop = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Test train_transform
    tensor_train = train_transform(dummy_crop)
    assert tensor_train.shape == (3, 224, 224), f"Wrong shape: {tensor_train.shape}"
    assert tensor_train.dtype == torch.float32, f"Wrong dtype: {tensor_train.dtype}"
    print(f"  train_transform: PIL (224,224) -> tensor {tuple(tensor_train.shape)}  OK")
    print(f"  train value range: [{tensor_train.min():.3f}, {tensor_train.max():.3f}]")

    # Test val_transform
    tensor_val = val_transform(dummy_crop)
    assert tensor_val.shape == (3, 224, 224), f"Wrong shape: {tensor_val.shape}"
    print(f"  val_transform:   PIL (224,224) -> tensor {tuple(tensor_val.shape)}  OK")

    tensor_val2 = val_transform(dummy_crop)
    assert torch.allclose(tensor_val, tensor_val2), "val_transform is not deterministic"
    print("  val_transform determinism: OK")

    tensor_train2 = train_transform(dummy_crop)
    assert tensor_train2.shape == (3, 224, 224)
    print("  train_transform second run: OK")

    # After Normalize, values are centred around 0, not in [0, 1]
    raw_tensor = transforms.ToTensor()(dummy_crop)   # [0, 1]
    norm_tensor = val_transform(dummy_crop)           # centred around 0
    assert raw_tensor.min() >= 0.0 and raw_tensor.max() <= 1.0
    assert norm_tensor.min() < 0.0, "Normalised tensor should have negative values"
    print("  Normalisation shifts values below 0 as expected: OK")

    print("\nAll assertions passed.")
