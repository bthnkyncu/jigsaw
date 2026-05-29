"""ORB feature matching score.

Brief §7.11: ORB descriptors + BFMatcher with Hamming distance + crossCheck;
score = fraction of matches whose distance is below ``orb_match_distance_max``.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from puzzle_assistant.config import Settings


def score_orb(
    piece_descriptors: np.ndarray | None,
    cell_descriptors: np.ndarray | None,
    settings: Settings,
) -> float:
    """Return ORB-feature score in ``[0, 1]``."""

    if piece_descriptors is None or cell_descriptors is None:
        return 0.0
    if piece_descriptors.shape[0] == 0 or cell_descriptors.shape[0] == 0:
        return 0.0

    bf: Any = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(piece_descriptors, cell_descriptors)
    except cv2.error:
        return 0.0
    if not matches:
        return 0.0
    good = sum(1 for m in matches if m.distance < settings.orb_match_distance_max)
    return float(good) / float(len(matches))


def compute_descriptors(image_bgr: np.ndarray, settings: Settings) -> np.ndarray | None:
    """Compute ORB descriptors for an arbitrary BGR image."""

    if image_bgr.size == 0:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb: Any = cv2.ORB_create(nfeatures=settings.orb_n_features)  # type: ignore[attr-defined]
    _kp, des = orb.detectAndCompute(gray, None)
    if des is None or des.shape[0] == 0:
        return None
    result: np.ndarray = des
    return result
