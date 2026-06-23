"""Combine the three sub-scores and apply the margin rule.

Brief §7.11: ``combined = TEMPLATE_WEIGHT * t + FEATURE_WEIGHT * f + COLOR_WEIGHT * c``;
the best cell wins iff ``combined >= MIN_COMBINED_SCORE`` and the gap to the
second-best cell (margin) is at least ``MIN_MARGIN``. When the target map is
``fallback`` quality, the thresholds become ``FALLBACK_*`` (stricter).
"""

from __future__ import annotations

from dataclasses import dataclass

from puzzle_assistant.config import Settings
from puzzle_assistant.reference.target_map import TargetMap
from puzzle_assistant.utils.coords import CellAddress


@dataclass
class CellScore:
    cell: CellAddress
    template: float
    feature: float
    color: float
    combined: float


@dataclass
class MatchResult:
    cell: CellAddress | None
    combined: float
    margin: float
    rejected_reason: str | None  # None on accept, e.g. "low_score" / "low_margin"
    texture: float = 0.0  # foreground texture (std-dev) of the picked piece
    # Diagnostics: the runner-up candidate's cell (reveals distant-twin ties on
    # rejections) and the seam continuity score when the tie-breaker resolved a
    # repeated-texture tie. Both purely informational for the match log.
    runner_up: tuple[int, int] | None = None
    seam_score: float | None = None


def combine_score(t: float, f: float, c: float, settings: Settings) -> float:
    return (
        settings.template_weight * t
        + settings.feature_weight * f
        + settings.color_weight * c
    )


def decide(scores: list[CellScore], target_map: TargetMap, settings: Settings) -> MatchResult:
    """Pick the winning cell after applying the margin rule."""

    if not scores:
        return MatchResult(cell=None, combined=0.0, margin=0.0, rejected_reason="empty")

    sorted_scores = sorted(scores, key=lambda s: s.combined, reverse=True)
    best = sorted_scores[0]
    second_best = sorted_scores[1].combined if len(sorted_scores) > 1 else 0.0
    margin = best.combined - second_best

    if target_map.quality == "primary":
        min_combined = settings.min_combined_score
        min_margin = settings.min_margin
    else:
        min_combined = settings.fallback_min_combined_score
        min_margin = settings.fallback_min_margin

    if best.combined < min_combined:
        return MatchResult(
            cell=None, combined=best.combined, margin=margin, rejected_reason="low_score"
        )
    if margin < min_margin:
        return MatchResult(
            cell=None, combined=best.combined, margin=margin, rejected_reason="low_margin"
        )
    return MatchResult(
        cell=best.cell, combined=best.combined, margin=margin, rejected_reason=None
    )
