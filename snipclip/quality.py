"""Image quality assessment using numpy/opencv.

Scores: sharpness (Laplacian variance), brightness, contrast, blur detection.
Pure numpy + opencv — zero ML dependencies.
"""

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from snipclip._cv import imread


class QualityScore(NamedTuple):
    """Image quality metrics."""
    sharpness: float      # Laplacian variance (higher = sharper)
    brightness: float     # Mean pixel value (0-255)
    contrast: float       # Standard deviation of pixel values
    is_blurry: bool       # True if likely blurry
    overall: float        # Combined score 0.0-1.0


def assess_quality(image_path: Path) -> QualityScore:
    """Assess image quality from a file path.

    Args:
        image_path: Path to image file.

    Returns:
        QualityScore with all metrics.
    """
    img = imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return QualityScore(0.0, 0.0, 0.0, True, 0.0)

    return _assess_from_image(img)


def _assess_from_image(img: np.ndarray) -> QualityScore:
    """Assess quality from a numpy grayscale image."""
    # Laplacian variance — measures sharpness
    lap = cv2.Laplacian(img, cv2.CV_64F)
    sharpness = float(lap.var())

    # Brightness and contrast
    brightness = float(np.mean(img))
    contrast = float(np.std(img))

    # Blur detection: low sharpness + low contrast
    is_blurry = sharpness < 50 or (sharpness < 100 and contrast < 30)

    # Overall score (normalize sharpness to 0-1, cap at ~500 for typical videos)
    sharp_norm = min(sharpness / 500.0, 1.0)
    bright_score = 1.0 - abs(brightness - 128) / 128  # best around mid-brightness
    contrast_norm = min(contrast / 80.0, 1.0)

    overall = 0.5 * sharp_norm + 0.25 * bright_score + 0.25 * contrast_norm

    return QualityScore(
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        is_blurry=is_blurry,
        overall=round(overall, 3),
    )
