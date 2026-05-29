"""Lab color signature distance score.

Brief §7.11: mean Lab vector L2-distance, mapped to ``[0, 1]``
via ``score = max(0, 1 - dist / 50)``.
"""

from __future__ import annotations

import cv2
import numpy as np


def score_color(piece_target: np.ndarray, cell_lab_mean: np.ndarray) -> float:
    """Return color-mean similarity score in ``[0, 1]``."""

    if piece_target.size == 0:
        return 0.0
    lab = cv2.cvtColor(piece_target, cv2.COLOR_BGR2LAB)
    piece_mean = lab.reshape(-1, 3).mean(axis=0).astype(np.float32)
    dist = float(np.linalg.norm(piece_mean - cell_lab_mean))
    return max(0.0, 1.0 - dist / 50.0)
