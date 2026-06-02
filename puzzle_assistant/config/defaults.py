"""Runtime configuration constants. Brief §8.

User overrides may be provided via ``config/settings.json`` at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Settings:
    # --- Window ---
    game_window_title_substring: str = "YapBoz"
    # Screen dimensions are no longer asserted — the pipeline adapts to whatever
    # bbox the game window reports. ``expected_dpi_scale`` is still enforced
    # because non-1.0 scaling breaks pixel-precise overlay placement.
    expected_dpi_scale: float = 1.0

    # --- Main loop ---
    target_fps: int = 20
    window_bbox_refresh_every_n_frames: int = 10
    # Raised from 5 % — while a piece is dragged across the board edge the
    # foreground contour briefly swells, jittering the detected board bbox by
    # 10–20 %. A real board move (zoom/pan) shifts it far more, so 35 % avoids
    # spurious recalibration that would throw away a good reference.
    board_bbox_change_threshold_pct: float = 35.0

    # --- HSV masks (Brief §8) ---
    background_blue_hsv_low: tuple[int, int, int] = (95, 60, 100)
    background_blue_hsv_high: tuple[int, int, int] = (125, 200, 220)
    board_light_hsv_low: tuple[int, int, int] = (0, 0, 200)
    board_light_hsv_high: tuple[int, int, int] = (180, 40, 255)

    # --- Init view watcher ---
    # Lowered from 1500 — board crop sometimes includes desk pixels (variance
    # is brought down by ~half). Stability is the primary signal now.
    init_view_variance_min: float = 400.0
    # Histogram correlation between the board crop and the right-side reference
    # panel. On the true init view this is ~0.5; with scattered pieces it drops
    # to ~0.04. 0.30 cleanly separates the two (live-run measured).
    init_view_panel_corr_min: float = 0.30
    init_view_wait_timeout_s: float = 20.0
    # Mean per-pixel diff between consecutive 128×96 board crops counted as
    # "stable". Picked from Puzzle_Game prototype (mean_diff < 1.5 there, but
    # we resize down so a slightly looser bound is correct here).
    init_view_stable_diff_max: float = 2.5
    # Consecutive stable frames required to declare the init view captured.
    # At 20 FPS this is ~0.3s of motion-free board content.
    init_view_stable_frame_count: int = 6
    # The assembled puzzle fills in top-to-bottom over ~1s, so the first frame
    # that looks "stable + matching" can still have unsettled bottom rows. Once
    # all criteria first hold we wait this long (criteria must keep holding)
    # before capturing, so the board is fully settled.
    init_view_settle_delay_s: float = 1.0

    # --- Grid detection ---
    # Gamyun's "250 parça" mode actually ships boards from ~230 to ~260 pieces
    # depending on the source image aspect, so the band is wider than the
    # brief's 240–260.
    expected_piece_count_min: int = 200
    expected_piece_count_max: int = 320
    grid_peak_prominence: float = 0.15

    # --- Pickup & segmentation ---
    # Tightened from 1.5 — a single piece (tabs included) is ~1.4 cells wide,
    # so a 1.5-cell *radius* (3-cell window) pulled neighbouring pieces from a
    # pile into the crop and the segmenter returned a multi-piece blob. 1.1
    # keeps the window just big enough for one piece plus its tabs.
    cursor_capture_radius_cell_multiplier: float = 1.1
    min_piece_area_ratio: float = 0.4
    group_area_ratio: float = 4.0
    piece_core_erode_ratio: float = 0.15

    # --- Board state (empty-cell tracking) ---
    # While READY, the filled[][] matrix is refreshed at most this often
    # (wall-clock s) so the empty-cell filter has a current view without
    # re-scanning every frame.
    board_state_refresh_s: float = 0.5
    # A cell counts as filled once less than (1 - this) of it is bare board
    # light — i.e. a placed piece covers most of it. Live boards split cleanly
    # (empty cells ≈1.0 light fraction, filled ≈0.0), so 0.45 sits in the empty
    # gap and classifies robustly without falsely marking empty cells filled.
    empty_cell_min_content_ratio: float = 0.45

    # --- Matching (primary quality) ---
    # template/feature/color weights are retained for config compatibility but
    # the sliding-window matcher (matching.engine) blends CCORR + CCOEFF with
    # its own fixed weights; only the thresholds below apply to it.
    template_weight: float = 0.50
    feature_weight: float = 0.30
    color_weight: float = 0.20
    # Gate on the blended appearance score. Recalibrated for the ORB-as-bonus
    # blend (engine.match): with ORB no longer dead-weighting the score, a
    # correctly-localized piece now lands ~0.60–0.92 (was ~0.32–0.49), while
    # wrong-puzzle pieces land ~0.45. Measured separation: at 0.55 the live
    # captures keep ~87 % recall while ~87 % of cross-puzzle pieces are
    # rejected — the balance point for the ≥80 % predict / ~95 % accuracy goal.
    min_combined_score: float = 0.55
    # Raised from 0.02 — at 0.02 ambiguous (near-tie) matches were accepted and
    # placed in the wrong cell. A wrong overlay is worse than none, so require
    # a clearer lead over the runner-up.
    min_margin: float = 0.05

    # Pieces flatter than this foreground grayscale std-dev are treated as
    # "single-colour-ish": their location is ambiguous (the colour repeats
    # across the board), so they need a much larger margin to be trusted.
    piece_texture_flat_max: float = 18.0
    flat_piece_min_margin: float = 0.15

    # Flat-edge border constraint. A puzzle piece with a *straight* (flat) edge
    # must sit on the corresponding board border (flat top → row 0, flat left →
    # col 0, etc.); a corner piece has two flat edges. We read the flat edges
    # off the dragged piece's silhouette and drop candidate cells that aren't on
    # that border — like the board-state filter, this only *removes* candidates,
    # so it can never cause a wrong placement. It rescues uniform-coloured
    # border regions (sky/water) where appearance ties a border piece against
    # spurious interior peaks. An edge counts as flat when its silhouette
    # boundary profile (central band, 10–90th pct range) varies less than this
    # fraction of the piece size; a tab/blank bulges ~0.2–0.3, so 0.12 keeps a
    # wide safety gap and flat detection biases to false-negative (no harm).
    flat_edge_max_deviation: float = 0.12
    # The board reference (~709×501, cell ~42 px) is upscaled by this factor
    # before matching so CCOEFF and ORB have enough detail to separate
    # repeated-texture pieces. 1.5 → cell ~63 px, keeping P95 latency under the
    # 200 ms budget while still adding the detail ORB needs.
    match_upscale_factor: float = 1.5

    # --- Matching (fallback quality) ---
    # Bumped in step with the primary gate for the ORB-as-bonus score scale.
    fallback_min_combined_score: float = 0.65
    fallback_min_margin: float = 0.06

    # --- Overlay ---
    overlay_color: str = "#00FF00"
    overlay_border_px: int = 3
    overlay_fill_alpha: float = 0.30
    overlay_alpha_high: float = 0.85
    overlay_alpha_low: float = 0.40
    overlay_blink_interval_ms: int = 600

    # --- Ref panel sensor ---
    ref_panel_check_interval_s: float = 3.0
    ref_panel_change_threshold: float = 0.35

    # --- Watchdog ---
    iteration_warn_ms: int = 500
    iteration_abort_ms: int = 2000

    # --- Logging ---
    log_dir: str = "logs"
    log_filename: str = "puzzle_assistant.log"
    log_rotate_bytes: int = 5 * 1024 * 1024
    log_rotate_backups: int = 3

    # --- ORB ---
    orb_n_features: int = 500
    orb_match_distance_max: int = 50

    extra: dict[str, Any] = field(default_factory=dict)


def load_settings(override_path: Path | None = None) -> Settings:
    """Load defaults, then merge keys from optional ``settings.json``."""

    settings = Settings()
    if override_path is None:
        return settings
    if not override_path.exists():
        return settings

    raw: dict[str, Any] = json.loads(override_path.read_text(encoding="utf-8"))
    valid_names = {f.name for f in fields(settings)}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        if key in valid_names:
            setattr(settings, key, _coerce(getattr(settings, key), value))
        else:
            extra[key] = value
    settings.extra.update(extra)
    return settings


def _coerce(default_value: Any, raw_value: Any) -> Any:
    """Best-effort coercion of JSON values into the dataclass field type."""

    if isinstance(default_value, tuple) and isinstance(raw_value, list):
        return tuple(raw_value)
    return raw_value
