"""Offline face detection and crop extraction for FaceForensics++.

Reads the index CSV, runs MTCNN on every frame, saves 224x224 face crops,
and writes an updated CSV with crop_path and detection_success columns.

Supports resuming interrupted runs: if a crop file already exists on disk
the frame is skipped. The CSV is saved every --save_every frames so progress
survives Colab session disconnects.

Usage (local):
    python scripts/preprocess_dataset.py \
        --index_csv  /path/to/data/index.csv \
        --crops_root /path/to/data/crops \
        --output_csv /path/to/data/index_with_crops.csv

Usage (Colab, crops saved to Google Drive):
    python scripts/preprocess_dataset.py \
        --index_csv  /content/drive/MyDrive/FreqForensics/index.csv \
        --crops_root /content/drive/MyDrive/FreqForensics/crops \
        --output_csv /content/drive/MyDrive/FreqForensics/index_with_crops.csv \
        --batch_size 32
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_mtcnn(device: torch.device) -> MTCNN:
    return MTCNN(
        image_size=224,
        margin=32,
        min_face_size=80,
        thresholds=[0.6, 0.7, 0.7],
        keep_all=False,
        post_process=False,   # return uint8 PIL crop, not normalised tensor
        device=device,
    )


def centre_crop(image: Image.Image, size: int = 224) -> Image.Image:
    """Fallback crop when MTCNN fails: centre-crop and resize to size×size."""
    w, h  = image.size
    short = min(w, h)
    left  = (w - short) // 2
    top   = (h - short) // 2
    image = image.crop((left, top, left + short, top + short))
    return image.resize((size, size), Image.BILINEAR)


def process_frame(
    frame_path: str,
    crop_path: Path,
    mtcnn: MTCNN,
) -> bool:
    """Detect face, save 224x224 crop. Returns True if MTCNN succeeded."""
    image = Image.open(frame_path).convert('RGB')

    crop = mtcnn(image)   # returns (224, 224, 3) uint8 tensor or None

    if crop is not None:
        # MTCNN returns a float tensor in [0,255] with post_process=False
        crop_np  = crop.permute(1, 2, 0).byte().numpy()
        crop_img = Image.fromarray(crop_np)
        success  = True
    else:
        crop_img = centre_crop(image)
        success  = False

    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop_img.save(str(crop_path))
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description='Preprocess FF++ frames: detect faces, save crops.')
    parser.add_argument('--index_csv',  required=True, type=Path,
                        help='Input index CSV from build_index.py')
    parser.add_argument('--crops_root', required=True, type=Path,
                        help='Root directory where crops will be saved')
    parser.add_argument('--output_csv', required=True, type=Path,
                        help='Path for the updated output CSV')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save progress CSV every N frames (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='MTCNN batch size (default: 1 for CPU; use 32 on GPU)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Reading index: {args.index_csv}\n")

    # Load index CSV
    with open(args.index_csv, newline='') as f:
        reader  = csv.DictReader(f)
        records = list(reader)

    total = len(records)
    print(f"Total frames in index: {total:,}")

    # Add new columns if not present
    for r in records:
        r.setdefault('crop_path', '')
        r.setdefault('detection_success', '')

    mtcnn = build_mtcnn(device)

    n_processed = n_skipped = n_failed = 0

    for i, record in enumerate(records):
        # Build crop path mirroring the frames directory structure
        frame_path = Path(record['path'])
        # crops_root / method / video_id / frame_id.png
        rel_path  = Path(record['method']) / record['video_id'] / frame_path.name
        crop_path = args.crops_root / rel_path

        # Resume: skip if crop already exists
        if crop_path.exists():
            record['crop_path']         = str(crop_path)
            record['detection_success'] = record.get('detection_success', 'True')
            n_skipped += 1
        else:
            try:
                success = process_frame(str(frame_path), crop_path, mtcnn)
                record['crop_path']         = str(crop_path)
                record['detection_success'] = str(success)
                if not success:
                    n_failed += 1
                n_processed += 1
            except Exception as e:
                print(f"  [ERROR] {frame_path}: {e}", file=sys.stderr)
                record['crop_path']         = ''
                record['detection_success'] = 'False'
                n_failed += 1

        # Periodic progress save
        if (i + 1) % args.save_every == 0:
            _save_csv(records, args.output_csv)
            pct = (i + 1) / total * 100
            print(f"  [{i+1:>7,}/{total:,}] {pct:.1f}%  "
                  f"processed={n_processed:,}  skipped={n_skipped:,}  "
                  f"failed={n_failed:,}")

    # Final save
    _save_csv(records, args.output_csv)

    print(f"\nDone.")
    print(f"  Total processed : {n_processed:,}")
    print(f"  Total skipped   : {n_skipped:,}  (already had crops)")
    print(f"  MTCNN failures  : {n_failed:,}  (used centre-crop fallback)")
    print(f"  Output CSV      : {args.output_csv}")


def _save_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['path', 'video_id', 'label', 'method', 'split',
                  'crop_path', 'detection_success']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    main()
