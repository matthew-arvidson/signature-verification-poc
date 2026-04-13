"""
Check signature extraction and normalization for offline verification.

Pipeline: load image → grayscale → optional ROI search → denoise →
background suppression → binarization → crop/pad → resize 224×224.

Designed for static check scans; heuristics assume the signature sits in the
lower portion of the image (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Target spatial size for Siamese / CNN backbones (e.g. ResNet-18).
TARGET_SIZE: Tuple[int, int] = (224, 224)


@dataclass
class PreprocessConfig:
    """Tunable parameters for check/signature preprocessing."""

    # Fraction of image height from the top to start searching for ink (0–1).
    signature_search_top: float = 0.35
    # Minimum contour area as a fraction of full image area.
    min_contour_area_frac: float = 0.0008
    # Maximum aspect ratio (width/height) for a valid signature box.
    max_aspect_ratio: float = 4.5
    # Padding around detected bounding box (fraction of box size).
    bbox_padding_frac: float = 0.12
    # Bilateral filter params for edge-preserving smoothing before thresholding.
    bilateral_d: int = 7
    bilateral_sigma_color: float = 55.0
    bilateral_sigma_space: float = 55.0
    # Morphological kernel sizes (odd integers).
    morph_close_ksize: Tuple[int, int] = (5, 5)
    morph_open_ksize: Tuple[int, int] = (3, 3)
    # Adaptive threshold block size (odd, > 1).
    adaptive_block_size: int = 21
    adaptive_C: int = 4


def load_image(
    path_or_array: Union[str, Path, np.ndarray],
    flags: int = cv2.IMREAD_COLOR,
) -> np.ndarray:
    """Load BGR image from path or pass through an ndarray."""
    if isinstance(path_or_array, (str, Path)):
        img = cv2.imread(str(path_or_array), flags)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
        return img
    if not isinstance(path_or_array, np.ndarray):
        raise TypeError("path_or_array must be a path or numpy array")
    return path_or_array.copy()


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA to single-channel uint8."""
    if bgr.ndim == 2:
        return bgr.astype(np.uint8)
    if bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def denoise_gray(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Edge-preserving smoothing to reduce scan noise before segmentation."""
    return cv2.bilateralFilter(
        gray,
        config.bilateral_d,
        config.bilateral_sigma_color,
        config.bilateral_sigma_space,
    )


def suppress_background_adaptive(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """
    Emphasize dark ink on light paper using adaptive threshold on inverted response.
    Returns a uint8 image where background is suppressed (lighter) and ink is darker.
    """
    blurred = denoise_gray(gray, config)
    # Inverted: ink becomes bright for adaptive threshold
    inverted = cv2.bitwise_not(blurred)
    th = cv2.adaptiveThreshold(
        inverted,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.adaptive_block_size,
        config.adaptive_C,
    )
    # th: bright where ink was strong — use as soft mask to attenuate background
    mask = th.astype(np.float32) / 255.0
    # Blend: keep original ink, fade regions that look like flat background
    blended = (gray.astype(np.float32) * (0.35 + 0.65 * mask)).astype(np.uint8)
    return blended


def binarize_signature(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """
    Binarize signature strokes: ink = 255, background = 0 (uint8).

    Uses Otsu on the background-suppressed image, then morphology to clean specks.
    """
    bg_suppressed = suppress_background_adaptive(gray, config)
    _, binary = cv2.threshold(
        bg_suppressed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Otsu on checks often yields dark background as high — ensure ink is white
    if float(np.mean(binary)) < 127:
        binary = cv2.bitwise_not(binary)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.morph_close_ksize)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.morph_open_ksize)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    return binary


def _largest_signature_bbox(
    binary_inv: np.ndarray,
    gray_shape: Tuple[int, int],
    config: PreprocessConfig,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find a bounding box around signature-like ink in the lower part of the check.

    `binary_inv` should have white ink on a black background. Rather than picking
    one contour, merge all meaningful stroke contours so disconnected letters are
    boxed as a single signature region.
    """
    h, w = gray_shape
    contours, _ = cv2.findContours(
        binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    y_min = int(h * config.signature_search_top)
    min_area = w * h * config.min_contour_area_frac
    candidate_boxes: list[Tuple[int, int, int, int]] = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if y + ch < y_min:
            continue
        area = cw * ch
        if area < min_area:
            continue
        candidate_boxes.append((x, y, cw, ch))

    if not candidate_boxes:
        return None

    x0 = min(x for x, _, _, _ in candidate_boxes)
    y0 = min(y for _, y, _, _ in candidate_boxes)
    x1 = max(x + cw for x, _, cw, _ in candidate_boxes)
    y1 = max(y + ch for _, y, _, ch in candidate_boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def detect_signature_bbox(
    gray: np.ndarray, config: PreprocessConfig
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect a bounding box (x, y, w, h) for the signature region using binarized ink.
    """
    binary = binarize_signature(gray, config)
    binary_inv = cv2.bitwise_not(binary)
    return _largest_signature_bbox(binary_inv, gray.shape, config)


def pad_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    padding_frac: float,
) -> Tuple[int, int, int, int]:
    """Expand bbox by padding_frac of its size, clipped to image bounds."""
    x, y, w, h = bbox
    pad_x = int(w * padding_frac)
    pad_y = int(h * padding_frac)
    H, W = image_shape
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(W, x + w + pad_x)
    y1 = min(H, y + h + pad_y)
    return (x0, y0, x1 - x0, y1 - y0)


def crop_region(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    x, y, w, h = bbox
    return image[y : y + h, x : x + w].copy()


def resize_to_target(
    gray: np.ndarray,
    size: Tuple[int, int] = TARGET_SIZE,
    pad_value: int = 255,
) -> np.ndarray:
    """
    Letterbox-resize: preserve aspect ratio, pad with pad_value to reach `size`.
    """
    tw, th = size[1], size[0]
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image for resize")

    scale = min(tw / w, th / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.full((th, tw), pad_value, dtype=resized.dtype)
    y0 = (th - new_h) // 2
    x0 = (tw - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def normalize_intensity(gray: np.ndarray) -> np.ndarray:
    """Optional contrast stretch to full 0–255 range."""
    g = gray.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    g = (g - lo) / (hi - lo) * 255.0
    return np.clip(g, 0, 255).astype(np.uint8)


def preprocess_signature_pipeline(
    path_or_array: Union[str, Path, np.ndarray],
    config: Optional[PreprocessConfig] = None,
    return_binary: bool = False,
) -> np.ndarray:
    """
    Full pipeline: load → grayscale → detect ROI → background handling →
    binarize → crop → letterbox resize 224×224 → optional normalization.

    Args:
        path_or_array: File path or BGR ndarray.
        config: PreprocessConfig; defaults used if None.
        return_binary: If True, return binary 0/255; else uint8 grayscale suitable for CNN.

    Returns:
        2D array of shape TARGET_SIZE (224, 224), dtype uint8.
    """
    cfg = config or PreprocessConfig()
    bgr = load_image(path_or_array)
    gray = to_grayscale(bgr)

    bbox = detect_signature_bbox(gray, cfg)
    if bbox is None:
        # Fallback: use lower-central band typical of MICR/signature zone
        H, W = gray.shape
        y0 = int(H * 0.5)
        bbox = (int(W * 0.45), y0, int(W * 0.5), H - y0)

    bbox = pad_bbox(bbox, gray.shape, cfg.bbox_padding_frac)
    crop_gray = crop_region(gray, bbox)

    # Background suppression + binarization on crop
    binary = binarize_signature(crop_gray, cfg)
    if return_binary:
        return resize_to_target(binary, TARGET_SIZE, pad_value=0)

    # For embedding networks, use clean grayscale: ink emphasized, background light
    bg_sup = suppress_background_adaptive(crop_gray, cfg)
    # Apply ink mask softly
    ink_mask = (binary > 127).astype(np.float32)
    ink_mask = cv2.GaussianBlur(ink_mask, (3, 3), 0)
    enhanced = (bg_sup.astype(np.float32) * (0.25 + 0.75 * ink_mask)).astype(np.uint8)
    enhanced = normalize_intensity(enhanced)

    # Final light denoise on resized image
    out = resize_to_target(enhanced, TARGET_SIZE)
    out = cv2.fastNlMeansDenoising(out, h=6, templateWindowSize=7, searchWindowSize=15)
    return out


def to_model_input_hwc(
    gray_224: np.ndarray,
    num_channels: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert 224×224 grayscale to H×W×C float32 array (e.g. 3-channel for ImageNet backbones).

    If normalize is True, scales to [0, 1]. Otherwise returns uint8 stacked.
    """
    if gray_224.ndim != 2:
        raise ValueError("Expected H×W grayscale")
    if num_channels == 1:
        g = gray_224.astype(np.float32) / 255.0 if normalize else gray_224
        return g[..., np.newaxis]
    stack = np.stack([gray_224] * num_channels, axis=-1)
    if normalize:
        return stack.astype(np.float32) / 255.0
    return stack.astype(np.uint8)


def to_model_input_chw(
    gray_224: np.ndarray,
    num_channels: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """CHW float tensor array (no torch dependency here)."""
    hwc = to_model_input_hwc(gray_224, num_channels=num_channels, normalize=normalize)
    return np.transpose(hwc, (2, 0, 1))
