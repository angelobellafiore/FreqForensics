"""Grad-CAM visualisation for FreqForensics.

For each fake method and for real faces, picks N example images from the
test split and produces a grid showing:
  - Original face crop
  - Grad-CAM heatmap (jet colormap)
  - Heatmap overlaid on the face

Usage:
    python visualisation/gradcam_vis.py \
        --checkpoint best.pt \
        --crops_csv  data/index_with_crops.csv \
        --output_dir results/gradcam/ \
        [--n_examples 4] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from models.freqforensics import FreqForensics


# ---------------------------------------------------------------------------
# ImageNet denormalisation (to display the original image)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (3, H, W) tensor to a uint8 HWC numpy array."""
    img = tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Grad-CAM (mirrors trainer.py implementation)
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model: FreqForensics) -> None:
        self.model       = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        target = model.spatial_branch.backbone.conv_head
        target.register_forward_hook(self._save_activations)
        target.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _m, _i, output) -> None:
        self._activations = output

    def _save_gradients(self, _m, _gi, grad_output) -> None:
        self._gradients = grad_output[0]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, H, W) normalised CAM maps upsampled to input resolution."""
        self.model.zero_grad()
        logit = self.model(x)
        logit.sum().backward(retain_graph=True)

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self._activations).sum(dim=1)
        cam     = torch.clamp(cam, min=0)

        # Upsample to input image size
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(x.shape[2], x.shape[3]),
            mode='bilinear', align_corners=False,
        ).squeeze(1)

        # Normalise per image
        B        = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cam_min  = cam_flat.min(dim=1).values.view(B, 1, 1)
        cam_max  = cam_flat.max(dim=1).values.view(B, 1, 1)
        cam      = (cam - cam_min) / (cam_max - cam_min).clamp(min=1e-8)

        return cam.detach()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def cam_to_heatmap(cam: np.ndarray) -> np.ndarray:
    """Convert a (H, W) float CAM in [0,1] to a uint8 RGB heatmap."""
    rgba = cm.jet(cam)                          # (H, W, 4) float
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend heatmap over image."""
    return (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)


def plot_examples(
    images:     list[np.ndarray],   # list of (H, W, 3) uint8
    cams:       list[np.ndarray],   # list of (H, W) float in [0,1]
    title:      str,
    output_path: Path,
    pred_probs: list[float],
) -> None:
    n   = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Face crop', 'Grad-CAM', 'Overlay']
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=11, fontweight='bold')

    for row, (img, cam, prob) in enumerate(zip(images, cams, pred_probs)):
        heatmap  = cam_to_heatmap(cam)
        overlaid = overlay(img, heatmap)

        axes[row, 0].imshow(img)
        axes[row, 1].imshow(heatmap)
        axes[row, 2].imshow(overlaid)

        for col in range(3):
            axes[row, col].axis('off')

        # Predicted probability label on the face crop
        axes[row, 0].set_xlabel(f'P(fake)={prob:.2f}', fontsize=9)
        axes[row, 0].xaxis.set_label_position('top')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True, type=Path)
    p.add_argument('--crops_csv',   required=True, type=Path)
    p.add_argument('--output_dir',  default=Path('results/gradcam'), type=Path)
    p.add_argument('--n_examples',  default=4, type=int,
                   help='Number of examples per method (default: 4)')
    p.add_argument('--seed',        default=42, type=int)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load model ---
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FreqForensics().to(device)
    model.load_state_dict(state['model'])
    model.eval()    # eval mode for BatchNorm stability; hooks still fire in eval mode
    grad_cam = GradCAM(model)

    # --- Load index ---
    df = pd.read_csv(args.crops_csv)
    if 'crop_path' in df.columns:
        df['path'] = df['crop_path'].where(
            df['crop_path'].notna() & (df['crop_path'] != ''), df['path']
        )
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    rng     = np.random.default_rng(args.seed)
    methods = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

    for method in methods:
        method_df = test_df[test_df['method'] == method]
        sample_df = method_df.sample(
            n=min(args.n_examples, len(method_df)),
            random_state=int(rng.integers(0, 10000)),
        )

        images_np  = []
        cams_np    = []
        pred_probs = []

        for _, row in sample_df.iterrows():
            image_pil = Image.open(row['path']).convert('RGB')
            tensor    = val_transform(image_pil).unsqueeze(0).to(device)

            cam   = grad_cam(tensor)                          # (1, H, W)
            prob  = torch.sigmoid(model(tensor)).item()

            images_np.append(denormalise(tensor.squeeze(0).cpu()))
            cams_np.append(cam.squeeze(0).cpu().numpy())
            pred_probs.append(prob)

        label   = 'Real' if method == 'original' else f'Fake — {method}'
        outfile = args.output_dir / f'gradcam_{method}.png'

        plot_examples(images_np, cams_np, label, outfile, pred_probs)

    print(f"\nAll Grad-CAM figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
