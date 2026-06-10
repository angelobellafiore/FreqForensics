import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN


# Canonical landmark positions for a 224x224 aligned face crop.
# These fixed target coordinates define where each facial feature should
# appear after alignment — regardless of the original pose or scale.
# Values are derived from standard face alignment templates used in the
# deepfake detection literature (e.g. FF++, FaceForensics baselines).
CANONICAL_LANDMARKS = np.array([
    [70.7,  112.0],   # left eye
    [153.3, 112.0],   # right eye
    [112.0, 142.0],   # nose tip
    [80.0,  172.0],   # left mouth corner
    [144.0, 172.0],   # right mouth corner
], dtype=np.float32)


def build_mtcnn(device: str = 'cpu') -> MTCNN:
    """Instantiate MTCNN with custom parameters for face detection and alignment.

    Key choices:
      image_size=224  — output crop matches EfficientNet-B4 input size
      margin=32       — 32px context around the bounding box captures the blend
                        boundary region which contains critical manipulation artifacts
      min_face_size=80 — ignores tiny background faces in crowd or multi-person shots
      keep_all=False  — return only the largest/highest-confidence face per frame
    """
    # MTCNN crops the face region in order to run the landmark detection
    return MTCNN(
        image_size=224,
        margin=32,
        min_face_size=80,
        thresholds=[0.6, 0.7, 0.7],
        keep_all=False,
        post_process=False,   # return pixel values in [0, 255], not normalised. By default, facenet-pytorch normalises the crop to [-1, 1]. We don't want that: we apply our own ImageNet normalisation later. See _align_face()
        device=device,
    )


def _align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int = 224) -> Image.Image:
    """Apply a similarity transform to align detected landmarks to the canonical template.

    A similarity transform preserves shape — it only allows rotation, uniform
    scaling, and translation (no shear, no independent x/y scaling). This is
    the right choice for faces: we want to remove pose variation without
    distorting facial proportions.

    Args:
        image:      HxWx3 numpy array (RGB uint8)
        landmarks:  (5, 2) array of detected landmark coordinates (x, y)
        output_size: side length of the output square crop

    Returns:
        224x224 PIL Image (RGB)
    """
    # estimateAffinePartial2D finds the best-fit similarity transform mapping
    # detected landmarks to canonical positions. RANSAC makes it robust to
    # one or two landmark detection errors.
    transform_matrix, _ = cv2.estimateAffinePartial2D(
        landmarks,
        CANONICAL_LANDMARKS,
        method=cv2.RANSAC,
    )

    if transform_matrix is None:
        # Landmark fitting failed completely — fall back to centre crop
        return _centre_crop(Image.fromarray(image), output_size)

    # Apply the affine warp to the full image
    aligned = cv2.warpAffine(
        image,
        transform_matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,   # reflect instead of zero-pad at edges
    )

    return Image.fromarray(aligned)


def _centre_crop(image: Image.Image, output_size: int = 224) -> Image.Image:
    """Fallback: crop the centre of the frame and resize to output_size x output_size.

    Used when MTCNN fails to detect a face. The manipulation signal is often
    still present in the centre of the frame even without a clean face crop.
    """
    w, h = image.size
    short_side = min(w, h)

    left   = (w - short_side) // 2
    top    = (h - short_side) // 2
    right  = left + short_side
    bottom = top  + short_side

    crop = image.crop((left, top, right, bottom))
    return crop.resize((output_size, output_size), Image.BILINEAR)


def detect_and_align(
    image: Image.Image,
    mtcnn: MTCNN,
    output_size: int = 224,
) -> tuple[Image.Image, bool]:
    """Detect the largest face in an image, align it, and return a square crop.

    Args:
        image:       PIL Image (RGB) — one video frame
        mtcnn:       MTCNN instance from build_mtcnn()
        output_size: side length of the output crop (default 224)

    Returns:
        crop:              output_size x output_size PIL Image (RGB)
        detection_success: True if MTCNN found a face, False if fallback was used
    """
    img_array = np.array(image)

    # detect() returns bounding boxes and landmarks without cropping
    # landmarks shape: (1, 5, 2) if a face is found, None otherwise
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

    if landmarks is None or len(landmarks) == 0:
        return _centre_crop(image, output_size), False

    # landmarks[0] is the (5, 2) array for the highest-confidence face
    face_landmarks = landmarks[0].astype(np.float32)
    crop = _align_face(img_array, face_landmarks, output_size)

    return crop, True


if __name__ == '__main__':
    print("Running smoke test for face_detector.py...")

    # Create a blank 480x270 RGB image (typical video frame size)
    # MTCNN won't find a real face here, so this tests the fallback path
    dummy_frame = Image.fromarray(np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8))

    mtcnn = build_mtcnn(device='cpu')
    crop, success = detect_and_align(dummy_frame, mtcnn, output_size=224)

    assert isinstance(crop, Image.Image), "Output is not a PIL Image"
    assert crop.size == (224, 224), f"Expected (224, 224), got {crop.size}"
    assert crop.mode == 'RGB', f"Expected RGB, got {crop.mode}"
    print(f"  detect_and_align: {dummy_frame.size} frame -> {crop.size} crop  OK")
    print(f"  detection_success: {success} (expected False for random noise image)")

    # Test centre crop directly
    crop_cc = _centre_crop(dummy_frame, 224)
    assert crop_cc.size == (224, 224), f"Centre crop size wrong: {crop_cc.size}"
    print(f"  _centre_crop: {dummy_frame.size} -> {crop_cc.size}  OK")

    print("\nAll assertions passed.")
