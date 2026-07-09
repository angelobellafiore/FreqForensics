"""Extract frames from FF++ .mp4 videos at a fixed temporal stride.

Usage:
    python data/extract_frames.py \
        --videos_root /path/to/FaceForensics++/c23 \
        --frames_root /path/to/output/frames \
        [--stride 10]

Expected input layout:
    videos_root/
        original_sequences/videos/
            000.mp4, 001.mp4, ...
        manipulated_sequences/
            Deepfakes/videos/
                000_003.mp4, ...
            Face2Face/videos/
            FaceSwap/videos/
            NeuralTextures/videos/

Output layout:
    frames_root/
        original/{video_id}/000001.png ...
        Deepfakes/{video_id}/000001.png ...
        ...
"""

import argparse
import subprocess
from pathlib import Path


METHODS = {
    'original':       'original',
    'Deepfakes':      'Deepfakes',
    'Face2Face':      'Face2Face',
    'FaceSwap':       'FaceSwap',
    'NeuralTextures': 'NeuralTextures',
}


def extract_video(mp4_path: Path, out_dir: Path, stride: int) -> int:
    """Extract every <stride>-th frame from a single video.

    Skips extraction if out_dir already contains PNG files, allows safe
    re-runs after interruption without re-processing completed videos.

    Returns the number of frames extracted (0 if skipped).
    """
    if out_dir.exists() and any(out_dir.glob('*.png')):
        return 0  # already extracted

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg',
        '-i',     str(mp4_path),
        '-vf',    f'select=not(mod(n\\,{stride}))',
        '-vsync', 'vfr',
        '-q:v',   '2',
        str(out_dir / '%06d.png'),
        '-loglevel', 'error',   # suppress ffmpeg progress noise
    ]

    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob('*.png')))


def extract_method(
    method: str,
    videos_root: Path,
    frames_root: Path,
    stride: int,
) -> tuple[int, int, int]:
    """Extract frames for all videos of one method.

    Returns (n_processed, n_skipped, n_frames_total).
    """
    method_videos_dir = videos_root / METHODS[method]

    if not method_videos_dir.exists():
        print(f"  [WARN] Not found, skipping: {method_videos_dir}")
        return 0, 0, 0

    mp4_files = sorted(method_videos_dir.glob('*.mp4'))
    if not mp4_files:
        print(f"  [WARN] No .mp4 files found in {method_videos_dir}")
        return 0, 0, 0

    n_processed = n_skipped = n_frames = 0

    for mp4 in mp4_files:
        video_id = mp4.stem          # e.g. "000" or "000_003"
        out_dir  = frames_root / method / video_id

        frames = extract_video(mp4, out_dir, stride)
        if frames == 0:
            n_skipped += 1
        else:
            n_processed += 1
            n_frames    += frames

    return n_processed, n_skipped, n_frames


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract FF++ frames via ffmpeg.')
    parser.add_argument('--videos_root', required=True, type=Path,
                        help='Root of the FF++ c23 dataset')
    parser.add_argument('--frames_root', required=True, type=Path,
                        help='Output root for extracted frames')
    parser.add_argument('--stride', type=int, default=10,
                        help='Extract every N-th frame (default: 10)')
    args = parser.parse_args()

    print(f"Extracting FF++ frames")
    print(f"  videos_root : {args.videos_root}")
    print(f"  frames_root : {args.frames_root}")
    print(f"  stride      : {args.stride} (every {args.stride}th frame)\n")

    total_processed = total_skipped = total_frames = 0

    for method in METHODS:
        print(f"[{method}]")
        processed, skipped, frames = extract_method(
            method, args.videos_root, args.frames_root, args.stride
        )
        print(f"  processed: {processed} videos  |  "
              f"skipped (already done): {skipped}  |  "
              f"frames extracted: {frames}")
        total_processed += processed
        total_skipped   += skipped
        total_frames    += frames

    print(f"\nDone.")
    print(f"  Total videos processed : {total_processed}")
    print(f"  Total videos skipped   : {total_skipped}")
    print(f"  Total frames extracted : {total_frames:,}")
    print(f"  Output directory       : {args.frames_root}")


if __name__ == '__main__':
    main()
