"""Accept/reject threshold behaviour, locked to real ground-truth measurements.

``scored_pickups_v3.json`` holds the matcher's per-candidate scores for 96 real
pickups from a live 100-piece game, each tagged with the cell the piece
*physically landed in*. 84 of them carry the reliable ``diff`` label (board-state
before/after); the other 12 fall back to the mouse release point, which is only
approximate and lands on a neighbouring cell often enough that it must not be
used to judge precision.

These tests pin the gate settings to that evidence: relaxing the gates was worth
it only because it cost zero errors on the reliable subset.
"""

from __future__ import annotations

import json
from pathlib import Path

from puzzle_assistant.config import load_settings


def _accepts(sample: dict, settings) -> bool:
    """Replicate the accept/reject gates of ``engine._match``."""
    combined = sample["combined"]
    margin = sample["margin"]
    second = combined - margin
    lone = (
        second <= settings.lone_candidate_max_second
        and combined >= settings.lone_candidate_floor
    )
    if combined < settings.min_combined_score and not lone:
        return False
    min_margin = (
        settings.flat_piece_min_margin
        if sample["texture"] < settings.piece_texture_flat_max
        else settings.min_margin
    )
    return margin >= min_margin


def _load(fixtures_dir: Path, reliable_only: bool = True) -> list[dict]:
    rows = json.loads((fixtures_dir / "scored_pickups_v3.json").read_text())
    if reliable_only:
        rows = [r for r in rows if r["label_source"] == "diff"]
    return rows


def test_precision_on_reliable_labels(fixtures_dir: Path) -> None:
    """No wrong overlay on any reliably-labelled pickup.

    A wrong overlay is worse than no overlay, so this is the gate that must
    never regress: at the tuned settings the measured error count is zero.
    """
    settings = load_settings(None)
    rows = _load(fixtures_dir)
    accepted = [r for r in rows if _accepts(r, settings)]
    wrong = [r for r in accepted if r["top"] != r["actual"]]
    assert not wrong, f"{len(wrong)} wrong prediction(s): {[w['sample'] for w in wrong]}"


def test_recall_on_reliable_labels(fixtures_dir: Path) -> None:
    """Predict on at least 95 % of pickups — the agreed target.

    Measured 98.8 % at the tuned settings (83/84, which is the ceiling: exactly
    one sample has a wrong top candidate and no threshold can rescue it).
    """
    settings = load_settings(None)
    rows = _load(fixtures_dir)
    accepted = [r for r in rows if _accepts(r, settings)]
    recall = len(accepted) / len(rows)
    assert recall >= 0.95, f"recall {recall:.1%} below the 95 % target"


def test_tightening_gates_would_lose_recall(fixtures_dir: Path) -> None:
    """Guard the tuning rationale: the old gates really did cost predictions.

    If someone reverts to 0.55/0.05 this fails, documenting that the relaxation
    was evidence-based rather than a guess.
    """
    settings = load_settings(None)
    rows = _load(fixtures_dir)
    now = sum(1 for r in rows if _accepts(r, settings))

    old = load_settings(None)
    old.min_combined_score = 0.55
    old.min_margin = 0.05
    before = sum(1 for r in rows if _accepts(r, old))
    assert now > before, "relaxed gates should predict on strictly more pickups"
