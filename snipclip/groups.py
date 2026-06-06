"""Scene grouping using HSV color histogram similarity.

Groups visually similar frames together — identifies scene clusters
without any ML model. Pure numpy + opencv.
"""

from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
from snipclip._cv import imread


class SceneGroup(NamedTuple):
    """A group of visually similar frames."""
    group_id: int
    keyframes: List[Path]          # frame paths in this group
    representative: Path           # best-quality frame of the group
    avg_color: tuple                # mean BGR color


def compute_histogram(image_path: Path, bins: int = 32) -> np.ndarray:
    """Compute normalized HSV histogram for an image."""
    img = imread(image_path)
    if img is None:
        return np.zeros(bins * 3, dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    return hist.astype(np.float32)


def group_similar_frames(
    frame_paths: List[Path],
    similarity_threshold: float = 0.7,
) -> List[SceneGroup]:
    """Group frames into visually similar clusters.

    Args:
        frame_paths: List of paths to frame images.
        similarity_threshold: Correlation threshold for grouping (0-1).

    Returns:
        List of SceneGroup namedtuples.
    """
    if not frame_paths:
        return []

    # Compute histograms for all frames
    hists = [compute_histogram(p) for p in frame_paths]

    # Simple greedy clustering
    groups: List[List[int]] = []  # list of index lists

    for i in range(len(frame_paths)):
        assigned = False
        for g_idx, group in enumerate(groups):
            # Compare with first frame in the group
            corr = cv2.compareHist(hists[i], hists[group[0]], cv2.HISTCMP_CORREL)
            if corr > similarity_threshold:
                group.append(i)
                assigned = True
                break
        if not assigned:
            groups.append([i])

    # Build SceneGroup objects
    scene_groups: List[SceneGroup] = []
    for g_id, indices in enumerate(groups):
        # Representative: pick frame with highest variance (most detail)
        best_idx = indices[0]
        best_var = 0
        for idx in indices:
            img = imread(frame_paths[idx])
            if img is not None:
                var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                if var > best_var:
                    best_var = var
                    best_idx = idx

        # Average color of representative
        rep_img = imread(frame_paths[best_idx])
        avg_color = tuple(int(c) for c in cv2.mean(rep_img)[:3]) if rep_img is not None else (0, 0, 0)

        scene_groups.append(SceneGroup(
            group_id=g_id,
            keyframes=[frame_paths[i] for i in indices],
            representative=frame_paths[best_idx],
            avg_color=avg_color,
        ))

    return scene_groups


def describe_scene(avg_color: tuple) -> str:
    """Heuristic scene label based on average color."""
    b, g, r = avg_color
    brightness = (b + g + r) / 3

    if brightness < 60:
        return "暗光/夜景"
    if brightness > 200:
        return "明亮/户外"

    if g > r and g > b:
        return "自然/户外"
    if b > r and b > g:
        return "蓝色调/水边"
    if r > g and r > b and r > 150:
        return "暖色调/室内"

    return "普通场景"
