"""OpenCV helpers with UTF-8 path support for Windows.

cv2.imread() doesn't handle non-ASCII paths on Windows.
These wrappers use numpy to read bytes, then decode with OpenCV.
"""

from pathlib import Path

import cv2
import numpy as np


def imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image from a path with full UTF-8 support.

    Args:
        path: Image file path (supports Chinese/Unicode on Windows).
        flags: cv2.IMREAD_* flag (default: IMREAD_COLOR).

    Returns:
        numpy array or None if read fails.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite(path: Path, img: np.ndarray) -> bool:
    """Write an image to a path with full UTF-8 support."""
    try:
        ext = path.suffix
        _, buf = cv2.imencode(ext, img)
        buf.tofile(str(path))
        return True
    except Exception:
        return False
