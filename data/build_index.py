"""Build the master split index CSV for FaceForensics++.

Reads the official train/val/test JSON split files and the extracted frames
directory to produce a single CSV mapping every frame to its video_id, label,
method, and split.

Usage:
    python data/build_index.py \
        --splits_root /path/to/FaceForensics++_C23/splits \
        --frames_root /path/to/FreqForensics/data/frames \
        --output      /path/to/FreqForensics/data/index.csv

The CSV has columns:
    path        absolute path to the extracted frame PNG
    video_id    e.g. "000" (real) or "000_003" (fake)
    label       0 for real, 1 for fake
    method      original / Deepfakes / Face2Face / FaceSwap / NeuralTextures
    split       train / val / test
"""

import argparse
import json
import csv
from pathlib import Path


FAKE_METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def load_split_lookup(splits_root: Path) -> dict[str, str]:
    """Build a {video_id: split} lookup from the three official JSON files.

    Each JSON contains a list of [id_a, id_b] pairs for fake videos, or
    plain id strings for real videos. We normalise both to a single
    video_id string: "000" for real, "000_003" for fake.
    """
    lookup = {}
    for split_name in ('train', 'val', 'test'):
        json_path = splits_root / f'{split_name}.json'
        with open(json_path) as f:
            entries = json.load(f)
        for entry in entries:
            # Entry is either a list ["000", "003"] or a plain string "000"
            if isinstance(entry, list):
                # Add both swap directions (e.g. "000_003" and "003_000")
                lookup[f'{entry[0]}_{entry[1]}'] = split_name
                lookup[f'{entry[1]}_{entry[0]}'] = split_name
                # Also add both source IDs for real video lookup (e.g. "000", "003")
                lookup[entry[0]] = split_name
                lookup[entry[1]] = split_name
            else:
                lookup[str(entry)] = split_name
    return lookup


def build_index(
    splits_root: Path,
    frames_root: Path,
    output_path: Path,
) -> None:
    print("Loading split JSON files...")
    split_lookup = load_split_lookup(splits_root)
    print(f"  Loaded {len(split_lookup)} video IDs across train/val/test\n")

    records = []
    missing_split = []

    # --- Real frames ---
    original_dir = frames_root / 'original'
    for video_dir in sorted(original_dir.iterdir()):
        video_id = video_dir.name
        split    = split_lookup.get(video_id)
        if split is None:
            missing_split.append(('original', video_id))
            continue
        for frame_path in sorted(video_dir.glob('*.png')):
            records.append({
                'path':     str(frame_path),
                'video_id': video_id,
                'label':    0,
                'method':   'original',
                'split':    split,
            })

    # --- Fake frames ---
    for method in FAKE_METHODS:
        method_dir = frames_root / method
        if not method_dir.exists():
            print(f"  [WARN] Method directory not found, skipping: {method_dir}")
            continue
        for video_dir in sorted(method_dir.iterdir()):
            video_id = video_dir.name
            split    = split_lookup.get(video_id)
            if split is None:
                missing_split.append((method, video_id))
                continue
            for frame_path in sorted(video_dir.glob('*.png')):
                records.append({
                    'path':     str(frame_path),
                    'video_id': video_id,
                    'label':    1,
                    'method':   method,
                    'split':    split,
                })

    print(f"Total frames indexed: {len(records):,}")

    if missing_split:
        print(f"\n[WARN] {len(missing_split)} video(s) had no matching split entry:")
        for method, vid in missing_split[:10]:
            print(f"  {method}/{vid}")
        if len(missing_split) > 10:
            print(f"  ... and {len(missing_split) - 10} more")

    # --- Leakage check ---
    print("\nRunning leakage check...")
    train_ids = {r['video_id'] for r in records if r['split'] == 'train'}
    test_ids  = {r['video_id'] for r in records if r['split'] == 'test'}
    overlap   = train_ids & test_ids
    assert not overlap, (
        f"DATA LEAK: {len(overlap)} video_id(s) appear in both train and test: "
        f"{list(overlap)[:5]}"
    )
    print("  No leakage detected — train and test video IDs are disjoint  OK")

    # --- Split summary ---
    for split_name in ('train', 'val', 'test'):
        n = sum(1 for r in records if r['split'] == split_name)
        print(f"  {split_name:5s}: {n:>8,} frames")

    # --- Write CSV ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['path', 'video_id', 'label', 'method', 'split']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nIndex written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Build FF++ split index CSV.')
    parser.add_argument('--splits_root', required=True, type=Path,
                        help='Folder containing train.json, val.json, test.json')
    parser.add_argument('--frames_root', required=True, type=Path,
                        help='Root of extracted frames (output of extract_frames.py)')
    parser.add_argument('--output', required=True, type=Path,
                        help='Path for the output index CSV')
    args = parser.parse_args()

    build_index(args.splits_root, args.frames_root, args.output)


if __name__ == '__main__':
    main()
