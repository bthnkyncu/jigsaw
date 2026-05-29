"""Template matching score (normalised cross-correlation).

Brief §7.11: ``cv2.matchTemplate(cell, piece, TM_CCOEFF_NORMED)`` produces a
single value in ``[-1, 1]``. We remap it into ``[0, 1]``.
"""

from __future__ import annotations

import cv2
import numpy as np


def score_template(piece_target: np.ndarray, cell_image: np.ndarray) -> float:
    """Return template-matching score in ``[0, 1]``.

    Both inputs must already share the same shape (the caller resizes
    ``piece_target`` to ``(cell_w, cell_h)`` before calling).
    """

    if piece_target.size == 0 or cell_image.size == 0:
        return 0.0
    if piece_target.shape != cell_image.shape:
        # Resize ``piece_target`` to match ``cell_image`` as a safety net.
        h, w = cell_image.shape[:2]
        piece_target = cv2.resize(piece_target, (w, h), interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(cell_image, piece_target, cv2.TM_CCOEFF_NORMED)
    raw = float(result[0, 0])
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))
