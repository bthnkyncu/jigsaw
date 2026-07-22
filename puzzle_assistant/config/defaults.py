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
    # The init view is the *assembled* puzzle, so the board must be covered in
    # picture — not showing through. Before the game starts it displays a
    # countdown banner over an otherwise bare board, which is motionless and
    # passes every other test; that screen got captured as the reference and the
    # whole game ran against it. Measured bare-board share: 0.4-10.3 % on the
    # references that were captured correctly (26.6 % on the hardest pale-blue
    # puzzle) versus ~85 % on the countdown screen. 40 % sits in that gap. This
    # also rejects a half-assembled board, where the unfilled part still shows.
    init_view_max_bare_board: float = 0.40
    init_view_wait_timeout_s: float = 20.0
    # Mean per-pixel diff between consecutive 128×96 board crops counted as
    # "stable". Picked from Puzzle_Game prototype (mean_diff < 1.5 there, but
    # we resize down so a slightly looser bound is correct here).
    init_view_stable_diff_max: float = 2.5
    # Consecutive stable frames required to declare the init view captured.
    # At 20 FPS this is ~0.3s of motion-free board content.
    init_view_stable_frame_count: int = 6
    # Extra motionless frames demanded when the reference panel can't be found.
    # Was 4 while stability was the only guard left in that case; the coverage
    # test below now rejects the countdown, the bare board and a half-built
    # board directly, so this stacked on top and simply made capture
    # impossible — the assembled view is only shown for ~1.5 s and needs to be
    # caught within it. Kept as a knob, but no longer a multiplier by default.
    init_view_no_panel_stable_multiplier: int = 1
    # The assembled puzzle fills in top-to-bottom over ~1s, so the first frame
    # that looks "stable + matching" can still have unsettled bottom rows. Once
    # all criteria first hold we wait this long (criteria must keep holding)
    # before capturing, so the board is fully settled.
    init_view_settle_delay_s: float = 1.0

    # --- Grid detection (parametric — NO hardcoded piece count) ---
    # The grid (rows × cols) is read from the cut-line periodicity, so it is
    # independent of zoom and piece count: any of 30 / 50 / 100 / 150 / 250 …
    # works. The bounds below are absolute cell sizes in *board pixels* — a
    # jigsaw cell is never smaller/larger than this regardless of how many
    # pieces. Cells are ~square, so the row and col periods are chosen to keep
    # cell_w/cell_h within ``grid_cell_aspect_max``; that squareness constraint
    # defeats the autocorrelation half/double-period harmonic that otherwise
    # doubled the row count on low-piece boards (the squashed overlay box).
    grid_min_cell_px: float = 24.0
    grid_max_cell_px: float = 230.0
    grid_cell_aspect_max: float = 1.5
    # How hard to penalise non-square cells when scoring candidate grids. The
    # aspect gate above has to stay loose (some boards really are oblong), so
    # squareness enters as a cost instead: a jigsaw cell is near-square, and an
    # oblong candidate winning means the cut-line signal was too weak to trust.
    grid_squareness_weight: float = 0.5
    grid_max_total_pieces: int = 520
    # Optional anchor: the piece count the user picked in the game (entered in
    # the GUI before "Başlat"). When set, grid detection only considers
    # rows×cols whose total is near this value, which eliminates the octave
    # errors (half/double/third periods) that pure pixel autodetection hits on
    # content-heavy images — the count moves the wrong total ~4× away. None ⇒
    # fall back to unanchored periodicity detection.
    target_piece_count: int | None = None
    # Tolerance band around target_piece_count (the game's actual count drifts a
    # few pieces from the nominal, and factorisations are sparse).
    target_piece_count_tolerance: float = 0.18
    # Retained for config/back-compat; no longer gates grid detection.
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
    # Desk (background) detection for piece segmentation. The fixed
    # ``background_blue_hsv_*`` band is far too wide: a puzzle whose image
    # contains sky/water sits in the SAME hue band as the desk, so 42 % of a
    # sky-coloured piece was being erased as "background" and the matcher
    # received a sliver (measured on a real capture: piece came out 17 px tall
    # instead of ~68). The desk is rendered as ONE exact colour (measured std =
    # 0 across the whole desk), while real image content always carries
    # texture — so we measure the desk colour from the crop border and match it
    # tightly instead. At tolerance 8 only ~9 % of sky pixels look like desk,
    # and the flatness test below removes almost all of those.
    desk_colour_tolerance: int = 8
    # A pixel only counts as desk if its local neighbourhood is flat. The desk
    # measures 0.0 local std; puzzle content medians 6.3. Being strict here is
    # the safe direction: it can only *keep* piece pixels, never erase them.
    desk_flat_std_max: float = 3.0
    # Fraction of the crop border that must match the modal border colour for
    # it to be trusted as "the piece sits on open desk". Below this (e.g. the
    # piece is picked straight out of an overlapping pile) we fall back to the
    # legacy fixed hue band.
    desk_border_uniform_min: float = 0.45

    # --- Board state (empty-cell tracking) ---
    # While READY, the filled[][] matrix is refreshed at most this often
    # (wall-clock s) so the empty-cell filter has a current view without
    # re-scanning every frame.
    board_state_refresh_s: float = 0.5
    # Self-supervised accuracy capture: after a piece is dropped, wait this long
    # for the game to snap/render, then diff the board to find where it actually
    # landed and log an `eval` event (predicted vs actual). Ground truth without
    # manual labelling — used to measure precision/recall and validate changes.
    eval_settle_s: float = 1.0
    # Board-bbox drift check (detect window move/zoom) runs at most this often.
    # detect_board on a full 2560×1440 capture is expensive; running it every
    # tick (20 FPS) floods board_detect_no_valid_contour and pegs a CPU core,
    # which made the mouse feel laggy on Windows. ~1.5 s is plenty to catch a
    # deliberate window move, and it now runs only while READY (never mid-drag).
    board_bbox_check_interval_s: float = 1.5
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
    # Held at 0.55. A sweep on 84 ground-truth samples showed 0.55, 0.50 and
    # 0.45 all accept exactly the same 80 pickups — the score gate was never
    # the binding constraint here, so lowering it buys nothing and only widens
    # the door for genuinely dim matches.
    min_combined_score: float = 0.55
    # Raised from 0.02 — at 0.02 ambiguous (near-tie) matches were accepted and
    # placed in the wrong cell. A wrong overlay is worse than none, so require
    # a clearer lead over the runner-up.
    #
    # Briefly relaxed to 0.03 because a sweep over 84 ground-truth samples said
    # it would add 3 correct predictions at zero cost. The very next live game
    # produced a wrong overlay at margin 0.034 — a failure mode absent from the
    # tuning set. That is overfitting to 84 samples, so it is back at 0.05, the
    # value two full games validated. Recall on the sweep drops 98.8 % -> 95.2 %,
    # still meeting the 95 % target, and precision is what the user actually
    # cares about. Do not lower this again without held-out evidence.
    min_margin: float = 0.05

    # Pieces flatter than this foreground grayscale std-dev are treated as
    # "single-colour-ish": their location is ambiguous (the colour repeats
    # across the board), so they need a much larger margin to be trusted.
    piece_texture_flat_max: float = 18.0
    flat_piece_min_margin: float = 0.15

    # Lone-candidate rescue. When the runner-up is negligible (the matcher found
    # ONE dominant peak and nothing else), the piece can only go there, so accept
    # it below the normal score gate. This is deliberately gated on the
    # SECOND-best being ~0 — not on a large margin — so it cannot fire on
    # repeated-texture ties (which have a real competing twin at half strength,
    # the failure mode that made the old confident-override hurt precision).
    lone_candidate_max_second: float = 0.12
    lone_candidate_floor: float = 0.42

    # Filter-decided guard. The empty-cell and flat-edge filters only *remove*
    # candidates, which is safe in principle — but a removal also erases the
    # runner-up the margin is measured against, so a lone survivor inherits a
    # fabricated margin and sails through both gates. Live this turned one
    # mis-detected cell into a sink: several different pieces were all predicted
    # onto [3,5] at combined 0.42-0.48 with margin == combined, because it was
    # the only cell board-state believed was empty.
    #
    # When a twin is filtered out because it is genuinely occupied it scores
    # about the same as the survivor (they are twins), so the lead stays near
    # zero and the rescue the filter exists for still works. A large lead means
    # the filters threw away something much better than what remained — the
    # answer came from filtering rather than from appearance — so reject.
    # Measured: the live sink discarded a candidate 0.44 ahead of the survivor,
    # while legitimate twin removals sit well under 0.15.
    filter_discard_max_lead: float = 0.15

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
    # --- Endgame empty-cell search (engine._search_empty_cells) ---
    # Runs only where the matcher was about to give up, and only once the board
    # is nearly finished: score the piece against each still-open cell instead
    # of against the top-N peaks of the whole board. Swept over every
    # appearance-miss in twelve games, |empty| <= 12 with a margin of 0.12
    # fires 17 times with zero errors; the first error appears at |empty| <= 15
    # with a 0.10 margin, so this sits inside the clean plateau on both axes.
    empty_cell_search_max_cells: int = 12
    empty_cell_search_min_margin: float = 0.12

    # --- Endgame hole-shape rescue (matching/hole_shape.py) ---
    # Runs only when appearance produced NO prediction, so it can add an overlay
    # but never overturn a correct one. Measured on 43 recorded pickups the
    # matcher failed to predict: 13 rescued, 0 errors.
    # Above this many empty cells the holes merge into multi-cell regions and
    # shape carries no information (measured 0 % correct above ~30 empty).
    #
    # 12 was too tight and was leaving rescues on the table: a 200-piece game
    # stalled with 13-16 holes open, just outside it. 15 lets that endgame be
    # considered at all. Precision is governed by ``hole_shape_min_gap``, not by
    # this cap — at the gap now in force, 12 / 14 / 15 all rescue the same 16
    # pickups with no errors.
    hole_shape_max_empty_cells: int = 15
    # A hole must fit this much better than the runner-up.
    #
    # 0.05 looked free for a long time — zero errors over every game recorded up
    # to that point — and then a leaf-and-water-droplet puzzle broke it. Every
    # cell of that image looks like every other, so the player was left with a
    # long run of *adjacent* holes along the bottom row, and adjacent holes merge
    # into one region whose outline inside any single cell is no longer a piece
    # silhouette. Shape then picks between neighbours almost at random: the
    # rescue supplied all four of that game's wrong overlays.
    #
    # Re-swept over every appearance-miss in twelve games, the one surviving
    # error is stubborn — no cap and no absolute-IoU floor removes it, only a
    # wider gap does. 0.12 is the first value that reaches zero errors, at a
    # real cost: 24 rescues -> 16. Taken deliberately, because a wrong overlay is
    # worse than none.
    #
    # Merging adjacent holes was the obvious alternative and it was measured:
    # keeping only holes with four filled neighbours makes it far worse (11-15
    # errors), because the true cell is often *in* the merged region and
    # dropping it lets a wrong isolated hole win by elimination — the same
    # filter-decides trap as [[filter-decided-sink]].
    hole_shape_min_gap: float = 0.12
    # Alignment search radius, in board pixels. Not cosmetic: without it the
    # same rule made 2 errors at gap 0.05, with it none.
    hole_shape_align_radius: int = 8
    # Fraction of a cell that must be bare board for it to count as a hole.
    hole_shape_empty_min: float = 0.45

    # Silhouette neutralisation. The tight piece crop is a rectangle, so the gaps
    # between the tabs hold desk, not puzzle content — and CCOEFF takes no mask,
    # so that outline joins the template and the match partly scores the piece's
    # *shape* against board luminance structure. Unrelated pieces then pile onto
    # the same few high-contrast cells. Filling the background with the piece's
    # own mean fixes it (measured on 266 recorded pickups: 71 % -> 97 % correct).
    # Guarded by this threshold because the foreground mask reads the background
    # colour off the crop corners, so on pure-content images it mislabels content
    # as background and filling would destroy the template. Desk is flat (colour
    # std 4.3 measured); misread content is not (27.7).
    silhouette_bg_max_std: float = 12.0
    # The board reference (~709×501, cell ~42 px) is upscaled by this factor
    # before matching so CCOEFF and ORB have enough detail to separate
    # repeated-texture pieces. 1.5 → cell ~63 px, keeping P95 latency under the
    # 200 ms budget while still adding the detail ORB needs.
    match_upscale_factor: float = 1.5
    # Margin (in cells) added around the reference board before matching, filled
    # by replicating the edge pixels.
    #
    # This is what capped border pieces. A piece's template is its whole
    # silhouette — ~1.5 cells across once the tabs are included — so a piece
    # belonging to a border cell needs to sit partly *outside* the board
    # rectangle. matchTemplate cannot place a template past the image edge, so
    # the best it can do is slide the piece inward: measured ~7 px, about 16 %
    # of a cell. Compared against content that far off, the correlation at the
    # piece's own cell collapses — 0.393 versus 0.538 for interior pieces — and
    # in 34 % of border pickups some unrelated spot on the board outscored the
    # right one (interior: 5 %).
    #
    # Padding makes the true position reachable. Measured on 1178
    # ground-truth-labelled pickups the border peak recovers to 0.524 (interior
    # is 0.538, so the gap all but closes) and interior pieces are untouched —
    # identical scores, identical predictions. End to end that is border recall
    # 67.1 % -> 80.4 %, and it improved in all eleven recorded games.
    #
    # 0.35 cells is enough to clear the widest tab; 0.6 measured identically, so
    # the smaller pad is kept for speed.
    board_match_pad_cells: float = 0.35

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
