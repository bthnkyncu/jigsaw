"""Top-level matching pipeline — sliding-window template match.

The earlier cell-by-cell ensemble (resize piece to one cell, score it against
every cell's cached features) was unreliable in practice: a dragged piece
carries cursor pixels, motion blur and a black-masked background, so every
cell scored ~0.28 with near-zero margin and nothing was ever selected.

The Puzzle_Game prototype instead localizes the piece directly on the full
reference board with a masked normalized cross-correlation, which is what we
do here:

1. Compute a foreground mask of the piece (non-background pixels) and tight-
   crop the piece to that mask.
2. ``cv2.matchTemplate(board, piece, TM_CCORR_NORMED, mask=fg)`` finds where
   the piece sits on the board. CCORR alone is permissive on flat regions, so
   we also run ``TM_CCOEFF_NORMED`` (mean-subtracted) on the tight crop and
   blend the two scores at each candidate.
3. Take the top-N non-max-suppressed peaks, pick the best blended score, and
   convert its board pixel position into a grid ``CellAddress``.

The returned ``MatchResult`` keeps the same shape the rest of the system
expects (``cell`` / ``combined`` / ``margin`` / ``rejected_reason``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.matching.ensemble import MatchResult
from puzzle_assistant.reference.target_map import TargetMap
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import CellAddress, GridSpec

if TYPE_CHECKING:
    from puzzle_assistant.piece.board_state import BoardState

_FG_THRESHOLD = 35.0


def match_piece(
    piece_bgr: np.ndarray,
    target_map: TargetMap,
    settings: Settings,
    board_state: "BoardState | None" = None,
    clipped_sides: tuple[bool, bool, bool, bool] | None = None,
) -> MatchResult:
    """Localize ``piece_bgr`` on the reference board and return its cell.

    When ``board_state`` is given, candidate positions that map to an
    already-filled cell are dropped before the best/runner-up decision. The
    repeated bouquet image makes a piece tie across distant board positions;
    once one of those twins is filled, removing it leaves the true empty cell
    and the margin recovers.
    """

    started = time.monotonic()
    result = _match(piece_bgr, target_map, settings, board_state, clipped_sides)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    plog.event(
        "match",
        level=logging.INFO,
        cell=[result.cell.row, result.cell.col] if result.cell else None,
        combined=round(result.combined, 3),
        margin=round(result.margin, 3),
        texture=round(result.texture, 1),
        quality=target_map.quality,
        elapsed_ms=round(elapsed_ms, 1),
        rejected_reason=result.rejected_reason,
    )
    return result


@dataclass
class Prepared:
    """Everything derived from one piece + board that scoring needs.

    Split out of ``_match`` so the endgame assignment can score a piece against
    a set of cells using *exactly* the same numbers the live matcher uses,
    rather than a re-implementation that could drift away from it.
    """

    board: np.ndarray
    fg: np.ndarray
    ccoeff: np.ndarray
    ccorr: np.ndarray | None
    piece_lab: np.ndarray
    orb: object
    bf: object
    p_desc: np.ndarray | None
    settings: Settings
    texture: float
    clipped: tuple[bool, bool, bool, bool]
    pw: int
    ph: int
    pad: int
    scale: float

    def score_at(self, x: int, y: int, ccoeff_score: float) -> float:
        cr = 0.0
        if self.ccorr is not None and 0 <= y < self.ccorr.shape[0] and 0 <= x < self.ccorr.shape[1]:
            cr = float(self.ccorr[y, x])
        # Colour agreement between the piece and the board patch it would cover.
        patch = self.board[y:y + self.ph, x:x + self.pw]
        color_score = _color_agreement(self.piece_lab, patch, self.fg)
        orb_score = _orb_agreement(self.orb, self.bf, self.p_desc, patch, self.settings)
        # Base appearance score from the three signals that actually fire on a
        # dragged ~60 px puzzle piece. ORB used to be a fixed 0.30 weight, but
        # its measured median on these pieces is 0.0 (too few keypoints; the
        # cross-check + Hamming gate is too strict at this resolution), so that
        # 0.30 was dead weight dragging *every* combined score down ~30 % and
        # pushing correctly-localized pieces below the gate. ORB now only adds a
        # bonus when it genuinely fires (repeated-texture pieces — fur, petals),
        # so it can still break those ties without capping the common case.
        base = 0.60 * ccoeff_score + 0.25 * cr + 0.15 * color_score
        return min(1.0, base + 0.15 * orb_score)

    def best_in_cell(
        self, row: int, col: int, grid: GridSpec
    ) -> tuple[float, int, int] | None:
        """Best score reachable with the piece centred on cell ``(row, col)``.

        Half a cell of slack, because the silhouette's centre is offset from the
        cell centre by however far the tabs stick out.
        """
        height, width = self.ccoeff.shape
        rx = int(grid.cell_w * self.scale * 0.45)
        ry = int(grid.cell_h * self.scale * 0.45)
        x0 = round((col + 0.5) * grid.cell_w * self.scale + self.pad - self.pw / 2)
        y0 = round((row + 0.5) * grid.cell_h * self.scale + self.pad - self.ph / 2)
        xa, xb = max(0, x0 - rx), min(width, x0 + rx + 1)
        ya, yb = max(0, y0 - ry), min(height, y0 + ry + 1)
        if xb <= xa or yb <= ya:
            return None
        window = self.ccoeff[ya:yb, xa:xb]
        wy, wx = divmod(int(np.argmax(window)), window.shape[1])
        x, y = xa + wx, ya + wy
        return self.score_at(x, y, float(window[wy, wx])), x, y


def prepare(
    piece_bgr: np.ndarray,
    target_map: TargetMap,
    settings: Settings,
    clipped_sides: tuple[bool, bool, bool, bool] | None = None,
) -> Prepared | str:
    """Build a :class:`Prepared`, or return the rejection reason as a string."""
    if piece_bgr.size == 0:
        return "empty_piece"

    # Upscale the whole match (board + piece) so a tiny ~42 px cell becomes
    # ~84 px. Both CCOEFF and ORB are far more discriminating at higher
    # resolution, which is what separates repeated-texture pieces (fur stripes,
    # petals, bark) that otherwise tie at several board positions.
    scale = settings.match_upscale_factor
    board = target_map.board_image
    if scale > 1.0:
        board = cv2.resize(
            board, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    # Let the piece hang off the board edge. A border piece's template spans
    # ~1.5 cells (tabs included) but only ~1 cell of board is left beside it, so
    # without a margin matchTemplate has to slide it inward and the correlation
    # at its own cell collapses. See ``board_match_pad_cells``.
    pad = round(
        settings.board_match_pad_cells
        * min(target_map.grid.cell_w, target_map.grid.cell_h)
        * scale
    )
    if pad > 0:
        board = cv2.copyMakeBorder(
            board, pad, pad, pad, pad, cv2.BORDER_REPLICATE
        )
    bh, bw = board.shape[:2]

    # Foreground mask + tight crop to the piece's real silhouette. If the mask
    # comes back nearly empty (e.g. a blue sky piece that the background
    # heuristic mistakes for desk), fall back to treating the whole crop as
    # foreground rather than rejecting — the CCOEFF signal still localizes it.
    fg = _foreground_mask(piece_bgr)
    if float(fg.mean()) < 12.0:
        fg = np.full(piece_bgr.shape[:2], 255, dtype=np.uint8)
    cols_any = np.where(fg.max(axis=0) > 0)[0]
    rows_any = np.where(fg.max(axis=1) > 0)[0]
    if len(cols_any) < 4 or len(rows_any) < 4:
        return "empty_fg"
    cx0, cx1 = int(cols_any[0]), int(cols_any[-1]) + 1
    cy0, cy1 = int(rows_any[0]), int(rows_any[-1]) + 1
    # Which sides were cut off by the capture window. A cut-off side looks
    # straight by artefact, so flat detection must be suppressed there. This CANNOT be derived
    # from ``piece_bgr`` when the caller already tight-cropped it — every side
    # then touches its own border and the test says "all four clipped", which
    # silently disabled the flat-edge constraint entirely. Segmentation knows
    # the answer (it saw the piece inside the region), so it passes it in; the
    # local fallback stays for callers that hand over an uncropped region.
    if clipped_sides is not None:
        clipped = clipped_sides
    else:
        orig_h, orig_w = piece_bgr.shape[:2]
        clipped = (cy0 == 0, cy1 >= orig_h, cx0 == 0, cx1 >= orig_w)
    piece = piece_bgr[cy0:cy1, cx0:cx1]
    fg = fg[cy0:cy1, cx0:cx1]
    if scale > 1.0:
        piece = cv2.resize(
            piece, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        fg = cv2.resize(fg, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    ph, pw = piece.shape[:2]

    if pw > bw or ph > bh or pw < 8 or ph < 8:
        return "bad_piece_size"

    # Neutralise the silhouette before CCOEFF. The tight crop is a *rectangle*,
    # so everything between the tabs is desk, not puzzle content — and CCOEFF
    # takes no mask, so that jigsaw outline becomes part of the template. The
    # match then partly scores the piece's *shape* against the board's light/dark
    # structure instead of its content, which is why unrelated pieces all piled
    # onto the same few high-contrast cells. Filling the background with the
    # piece's own mean removes that structure while leaving the content intact:
    # measured on 266 recorded pickups, correct localisation went 71 % -> 97 %.
    # Only when that background really is desk, though: ``_foreground_mask``
    # estimates the background colour from the crop corners, so on a piece image
    # that is pure puzzle content (the self-match path) it mistakes content for
    # background and filling would erase the template. The desk is a single flat
    # colour, image content never is — measured background colour std is 4.3 on
    # real pieces versus 27.7 on content misread as background.
    piece_ccoeff = piece
    background = fg == 0
    if background.any() and (~background).any():
        bg_px = piece[background].reshape(-1, 3).astype(np.float32)
        if float(bg_px.std(axis=0).mean()) < settings.silhouette_bg_max_std:
            piece_ccoeff = piece.copy()
            piece_ccoeff[background] = piece[~background].reshape(-1, 3).mean(axis=0)

    # CCOEFF (mean-subtracted) is the primary localizer: it discriminates the
    # true position on a textured board far better than CCORR, which is
    # brightness-dominated and scores flat regions almost uniformly (this is
    # what produced near-zero margins in live runs).
    try:
        ccoeff = cv2.matchTemplate(board, piece_ccoeff, cv2.TM_CCOEFF_NORMED)
        ccoeff = np.clip(ccoeff, 0.0, 1.0)
    except cv2.error:
        return "ccoeff_failed"

    # Masked CCORR as a secondary colour-fidelity signal.
    try:
        ccorr = cv2.matchTemplate(board, piece, cv2.TM_CCORR_NORMED, mask=fg)
        ccorr = np.nan_to_num(ccorr, nan=0.0)
        ccorr = np.clip(ccorr, 0.0, 1.0)
    except cv2.error:
        ccorr = None

    # ORB descriptors of the piece (foreground only). Feature matching breaks
    # the repeated-texture ties that template/colour can't: even when fur or
    # petals look alike across the board, the local keypoint geometry differs,
    # so the true patch yields more good matches.
    orb = cv2.ORB_create(nfeatures=settings.orb_n_features)  # type: ignore[attr-defined]
    piece_gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    _p_kp, p_desc = orb.detectAndCompute(piece_gray, fg)

    return Prepared(
        board=board, fg=fg, ccoeff=ccoeff, ccorr=ccorr,
        piece_lab=_masked_lab_mean(piece, fg),
        orb=orb, bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True), p_desc=p_desc,
        settings=settings, texture=_piece_texture(piece, fg), clipped=clipped,
        pw=pw, ph=ph, pad=pad, scale=scale,
    )


def _match(
    piece_bgr: np.ndarray,
    target_map: TargetMap,
    settings: Settings,
    board_state: "BoardState | None" = None,
    clipped_sides: tuple[bool, bool, bool, bool] | None = None,
) -> MatchResult:
    prep = prepare(piece_bgr, target_map, settings, clipped_sides)
    if isinstance(prep, str):
        return MatchResult(cell=None, combined=0.0, margin=0.0, rejected_reason=prep)

    fg, ccoeff = prep.fg, prep.ccoeff
    pw, ph, pad, scale = prep.pw, prep.ph, prep.pad, prep.scale
    texture, clipped = prep.texture, prep.clipped
    score_at = prep.score_at

    candidates = _top_n_candidates(ccoeff, pw, ph, n=8, min_score=0.0)
    if not candidates:
        return MatchResult(
            cell=None, combined=0.0, margin=0.0, rejected_reason="no_candidate",
            texture=texture,
        )

    scored: list[tuple[int, int, float]] = [
        (x, y, score_at(x, y, cc)) for x, y, cc in candidates
    ]

    scored.sort(key=lambda s: s[2], reverse=True)
    # Keep the unfiltered field: the filters below remove rivals *by rule*, not
    # because the appearance was unambiguous, so "no runner-up left" must not be
    # read as "this piece can only go here".
    raw_scored = list(scored)

    # Empty-cell filter. A dragged piece can only land in an empty cell, so
    # drop any candidate whose cell is already filled. On the repeated bouquet
    # image a piece ties across distant positions; once one twin is placed,
    # removing it leaves the true empty cell and the margin recovers.
    #
    # When *every* candidate is filled the piece's real cell never entered the
    # top-N at all, so the best survivor is a cell we can see is occupied — a
    # prediction with no chance of being right. This used to fall through
    # unfiltered on the theory that the board-state might be stale; it isn't.
    # Measured across 791 ground-truth pickups from nine games, board-state
    # called the true (still empty) cell "filled" exactly zero times. The escape
    # hatch guarded nothing and cost accuracy: late in a 200-piece game the same
    # occupied cell was shown over and over for a corner piece.
    if board_state is not None:
        empty = [
            cand for cand in scored
            if not board_state.is_filled(
                *_xy_to_cell(cand[0], cand[1], pw, ph, scale, target_map, pad)
            )
        ]
        if not empty:
            bx, by, bc = scored[0]
            cell = _search_empty_cells(prep, board_state, target_map, settings)
            return MatchResult(
                cell=cell, combined=bc, margin=0.0,
                rejected_reason=None if cell else "all_filled", texture=texture,
                top_cell=_xy_to_cell(bx, by, pw, ph, scale, target_map, pad),
            )
        scored = empty

    # Flat-edge border constraint. A piece with a straight (flat) silhouette
    # edge can only sit on the matching board border, so drop candidates that
    # aren't on it. Like the empty-cell filter this only removes candidates
    # (never invents a match), so it cannot cause a wrong placement; if it would
    # empty the set the detection is untrusted and we fall through unfiltered.
    flats = _detect_flat_edges(fg, clipped, settings)
    if any(flats):
        on_border = [
            cand for cand in scored
            if _on_border(
                _xy_to_cell(cand[0], cand[1], pw, ph, scale, target_map, pad),
                flats, target_map,
            )
        ]
        if on_border:
            scored = on_border

    best_x, best_y, best_combined = scored[0]
    second = scored[1][2] if len(scored) > 1 else 0.0
    margin = best_combined - second

    # How strong was the best candidate the filters discarded? When a twin is
    # removed because it is genuinely occupied, it scores about the same as the
    # survivor, so this stays small — that is the case the empty-cell filter
    # exists to rescue. But when board-state wrongly calls a filled cell empty,
    # the *correct* candidates get filtered away and a far weaker one survives
    # by elimination, inheriting a fabricated margin (nothing left to compare
    # against). That is how one mislabelled cell became a sink that swallowed
    # piece after piece live. A large lead here means the answer was decided by
    # filtering, not by appearance, so it cannot be trusted.
    kept = {(c[0], c[1]) for c in scored}
    best_discarded = max(
        (c[2] for c in raw_scored if (c[0], c[1]) not in kept), default=0.0
    )
    discarded_lead = best_discarded - best_combined
    top_row, top_col = _xy_to_cell(best_x, best_y, pw, ph, scale, target_map, pad)
    top_cell = (top_row, top_col)

    min_combined = (
        settings.min_combined_score
        if target_map.quality == "primary"
        else settings.fallback_min_combined_score
    )
    min_margin = (
        settings.min_margin
        if target_map.quality == "primary"
        else settings.fallback_min_margin
    )

    # Content-aware margin gate. A low-texture / near-single-colour piece has
    # an inherently ambiguous location: the same colour repeats all over the
    # board, so several candidates score alike and the top peak is often the
    # wrong one. For such pieces we demand a much larger margin before trusting
    # the match (and otherwise reject — a missing overlay beats a wrong one).
    if texture < settings.piece_texture_flat_max:
        min_margin = max(min_margin, settings.flat_piece_min_margin)

    # Lone-candidate rescue: one dominant peak with a negligible runner-up means
    # the piece can only go to that cell, so accept it below the score gate.
    # Gated on second≈0 (not on margin), so a repeated-texture tie — which has a
    # real competing twin — can never trigger it.
    #
    # The runner-up here MUST come from the unfiltered field. Late in a game the
    # empty-cell filter drops nearly every candidate, so one survivor is left and
    # the filtered ``second`` collapses to 0 — which used to read as "unrivalled"
    # and waved a weak match past the score gate. Worse, if board-state wrongly
    # marks one filled cell empty, that cell is the sole survivor for *every*
    # piece and they all funnel into it (observed live: 4+ pieces predicted onto
    # [3,5] at combined ~0.42–0.48, each with margin == combined).
    raw_second = max(
        (c[2] for c in raw_scored if (c[0], c[1]) != (best_x, best_y)),
        default=0.0,
    )
    lone = (
        raw_second <= settings.lone_candidate_max_second
        and best_combined >= settings.lone_candidate_floor
    )

    def give_up(reason: str) -> MatchResult:
        """Reject — unless the shortlist of still-open cells settles it.

        The top-N peaks are taken over the *whole* board, so a piece whose cell
        never made that shortlist cannot be recovered by any filter: the answer
        simply is not in the set being filtered. Late in a game, though, only a
        handful of cells are still open, and asking "of those, which does this
        piece fit best?" is a different and much easier question. See
        ``_search_empty_cells``.
        """
        cell = _search_empty_cells(prep, board_state, target_map, settings)
        if cell is not None:
            return MatchResult(
                cell=cell, combined=best_combined, margin=margin,
                rejected_reason=None, texture=texture, top_cell=top_cell,
            )
        return MatchResult(
            cell=None, combined=best_combined, margin=margin,
            rejected_reason=reason, texture=texture, top_cell=top_cell,
        )

    if discarded_lead > settings.filter_discard_max_lead:
        return give_up("filter_decided")

    if best_combined < min_combined and not lone:
        return give_up("low_score")
    if margin < min_margin:
        return give_up("low_margin")

    return MatchResult(
        cell=CellAddress(row=top_row, col=top_col),
        combined=best_combined,
        margin=margin,
        rejected_reason=None,
        texture=texture,
        top_cell=top_cell,
    )


def _search_empty_cells(
    prep: Prepared,
    board_state: "BoardState | None",
    target_map: TargetMap,
    settings: Settings,
) -> CellAddress | None:
    """Which of the still-open cells does the piece fit best?

    The normal path takes the top-N correlation peaks over the whole board and
    then removes the ones on filled cells. That cannot help a piece whose own
    cell never made the shortlist — filtering only ever removes, so the answer
    has to already be in the set. Late in a game the question can be turned
    around: a dozen cells are still open, so score the piece at each of *them*
    directly and see whether one wins clearly.

    Measured only after the reference gained its padding (see
    ``board_match_pad_cells``) — before that this rescued nothing at all (0 of
    15 tries), because a border piece could not reach its own position in the
    first place. With padding it puts the right cell first on 65 of 105 pickups
    the matcher had given up on, and at the gates below it fires 17 times with
    no errors, 9 of those beyond what the hole-shape rescue already caught.

    It is not a re-ranker: it runs only where the answer was going to be "no
    prediction", so it can add an overlay but never overturn one.
    """
    if board_state is None:
        return None
    grid = target_map.grid
    empty = [
        (r, c)
        for r in range(grid.rows)
        for c in range(grid.cols)
        if not board_state.is_filled(r, c)
    ]
    if not empty or len(empty) > settings.empty_cell_search_max_cells:
        return None

    ranked: list[tuple[float, tuple[int, int]]] = []
    for row, col in empty:
        hit = prep.best_in_cell(row, col, grid)
        if hit is not None:
            ranked.append((hit[0], (row, col)))

    # A margin needs something to measure against. With one open cell there is
    # no rival, so every piece "wins" there by default — and if board-state has
    # that one cell wrong, every piece funnels into it. That is the sink from
    # ``tests/test_filter_sink.py``, reached by a different route. Requiring a
    # real runner-up costs 2 of 17 measured rescues and closes it.
    if len(ranked) < 2:
        return None
    ranked.sort(reverse=True)
    best, cell = ranked[0]
    if best - ranked[1][0] < settings.empty_cell_search_min_margin:
        return None
    return CellAddress(row=cell[0], col=cell[1])


def _xy_to_cell(
    x: int, y: int, pw: int, ph: int, scale: float, target_map: TargetMap, pad: int = 0
) -> tuple[int, int]:
    """Map a candidate's top-left (upscaled, padded-board) pixel to ``(row, col)``.

    ``pad`` is the margin added around the board before matching; subtracting it
    puts the candidate back in board coordinates. Getting this wrong shifts
    every prediction by a fraction of a cell, so it is not optional.
    """
    center_x = (x + pw / 2 - pad) / scale
    center_y = (y + ph / 2 - pad) / scale
    col = int(min(target_map.grid.cols - 1, max(0, center_x // target_map.grid.cell_w)))
    row = int(min(target_map.grid.rows - 1, max(0, center_y // target_map.grid.cell_h)))
    return row, col


# Central fraction of each edge sampled for flatness, avoiding the corners and
# any adjacent-edge tab that would contaminate the profile.
_FLAT_EDGE_BAND = 0.5


def _detect_flat_edges(
    fg: np.ndarray, clipped: tuple[bool, bool, bool, bool], settings: Settings
) -> tuple[bool, bool, bool, bool]:
    """Which of (top, bottom, left, right) silhouette edges are flat (straight).

    A flat edge means a straight board border; a tab/blank bulges or notches in
    its centre. We measure the boundary profile's 10–90th percentile range over
    the central band of the edge: flat → near-constant (small range), tab/blank
    → large range. Sides flagged ``clipped`` (silhouette ran into the crop
    border) are forced non-flat, since a clipped edge is straight by artefact.
    """
    h, w = fg.shape[:2]
    if h < 12 or w < 12:
        return (False, False, False, False)
    thresh = settings.flat_edge_max_deviation
    sides = ("top", "bottom", "left", "right")
    out = [
        (not clip) and _edge_is_flat(fg, side, thresh)
        for side, clip in zip(sides, clipped, strict=True)
    ]
    return (out[0], out[1], out[2], out[3])


def _edge_is_flat(fg: np.ndarray, side: str, thresh: float) -> bool:
    h, w = fg.shape[:2]
    profile: list[int] = []
    if side in ("top", "bottom"):
        lo, hi = int(w * (1 - _FLAT_EDGE_BAND) / 2), int(w * (1 + _FLAT_EDGE_BAND) / 2)
        for x in range(lo, max(lo + 1, hi)):
            idx = np.where(fg[:, x] > 0)[0]
            if idx.size:
                profile.append(int(idx[0]) if side == "top" else int(idx[-1]))
        extent = h
    else:
        lo, hi = int(h * (1 - _FLAT_EDGE_BAND) / 2), int(h * (1 + _FLAT_EDGE_BAND) / 2)
        for y in range(lo, max(lo + 1, hi)):
            idx = np.where(fg[y, :] > 0)[0]
            if idx.size:
                profile.append(int(idx[0]) if side == "left" else int(idx[-1]))
        extent = w
    if len(profile) < 3 or extent <= 0:
        return False
    arr = np.asarray(profile, dtype=np.float32)
    rng = float(np.percentile(arr, 90) - np.percentile(arr, 10))
    return bool(rng / float(extent) < thresh)


def _on_border(
    cell: tuple[int, int], flats: tuple[bool, bool, bool, bool], target_map: TargetMap
) -> bool:
    """True if ``cell`` lies on every border implied by the piece's flat edges."""
    row, col = cell
    top, bottom, left, right = flats
    rows, cols = target_map.grid.rows, target_map.grid.cols
    return (
        (not top or row == 0)
        and (not bottom or row == rows - 1)
        and (not left or col == 0)
        and (not right or col == cols - 1)
    )


def _orb_agreement(
    orb: object,
    bf: object,
    piece_desc: np.ndarray | None,
    patch_bgr: np.ndarray,
    settings: Settings,
) -> float:
    """Fraction of the piece's ORB descriptors that find a good (crossCheck,
    Hamming < threshold) match in the board patch. 0 when there are too few
    features to judge. This is the signal that breaks repeated-texture ties.
    """
    if piece_desc is None or len(piece_desc) < 4:
        return 0.0
    if patch_bgr.size == 0:
        return 0.0
    patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    _q_kp, q_desc = orb.detectAndCompute(patch_gray, None)  # type: ignore[attr-defined]
    if q_desc is None or len(q_desc) < 4:
        return 0.0
    try:
        matches = bf.match(piece_desc, q_desc)  # type: ignore[attr-defined]
    except cv2.error:
        return 0.0
    if not matches:
        return 0.0
    good = sum(1 for m in matches if m.distance < settings.orb_match_distance_max)
    # Normalize by the number of piece descriptors so a feature-rich piece
    # isn't unfairly favoured.
    return min(1.0, good / max(len(piece_desc), 1))


def _piece_texture(piece_bgr: np.ndarray, fg: np.ndarray) -> float:
    """Texture richness of the piece: std-dev of foreground grayscale.

    A flat / single-colour piece scores low (<~18) and is hard to localize
    reliably; a detailed piece scores high.
    """
    gray = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2GRAY)
    mask = fg > 0
    if not mask.any():
        return float(gray.std())
    return float(gray[mask].std())


def _masked_lab_mean(piece_bgr: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Mean Lab colour of the foreground pixels of ``piece_bgr``."""
    lab = cv2.cvtColor(piece_bgr, cv2.COLOR_BGR2LAB)
    mask = fg > 0
    mean = lab.reshape(-1, 3).mean(axis=0) if not mask.any() else lab[mask].mean(axis=0)
    return np.asarray(mean, dtype=np.float32)


def _color_agreement(piece_lab: np.ndarray, patch_bgr: np.ndarray, fg: np.ndarray) -> float:
    """1.0 when the board patch's mean colour matches the piece, decaying with
    Lab L2 distance. Foreground-masked so tabs/background don't skew it.
    """
    if patch_bgr.size == 0:
        return 0.0
    patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    if patch_lab.shape[:2] == fg.shape[:2]:
        mask = fg > 0
        patch_mean = (
            patch_lab[mask].mean(axis=0).astype(np.float32)
            if mask.any()
            else patch_lab.reshape(-1, 3).mean(axis=0).astype(np.float32)
        )
    else:
        patch_mean = patch_lab.reshape(-1, 3).mean(axis=0).astype(np.float32)
    dist = float(np.linalg.norm(piece_lab - patch_mean))
    # ~25 Lab units of difference halves the score.
    return max(0.0, 1.0 - dist / 50.0)


def _foreground_mask(piece_bgr: np.ndarray) -> np.ndarray:
    """Non-background mask via corner-sampled L2 distance, with HSV fallback.

    Adapted from Puzzle_Game's ``_compute_fg_mask``: sample the four corners
    (which a tight cursor crop almost always fills with desk/board), estimate
    the background colour, and threshold pixels by distance from it.
    """

    h, w = piece_bgr.shape[:2]
    if h < 10 or w < 10:
        return np.ones((h, w), dtype=np.uint8) * 255

    cs = max(4, min(h, w) // 10)
    corners = np.vstack([
        piece_bgr[:cs, :cs].reshape(-1, 3),
        piece_bgr[:cs, w - cs:].reshape(-1, 3),
        piece_bgr[h - cs:, :cs].reshape(-1, 3),
        piece_bgr[h - cs:, w - cs:].reshape(-1, 3),
    ]).astype(np.float32)
    corner_std = float(corners.std(axis=0).mean())
    if corner_std < 30.0:
        bg = np.median(corners, axis=0)
        diff = piece_bgr.astype(np.float32) - bg
        dist = np.sqrt((diff ** 2).sum(axis=2))
        fg = (dist >= _FG_THRESHOLD).astype(np.uint8) * 255
        if 3 < float(fg.mean()) < 252:
            return _clean(fg)

    # Fallback: blue-dominant pixels are background.
    b = piece_bgr[:, :, 0].astype(np.int16)
    r = piece_bgr[:, :, 2].astype(np.int16)
    is_bg = ((b - r) > 20) & (b > 90)
    fg = (~is_bg).astype(np.uint8) * 255
    return _clean(fg)


def _clean(mask: np.ndarray) -> np.ndarray:
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k3)
    return out


def _top_n_candidates(
    result_map: np.ndarray, piece_w: int, piece_h: int, n: int, min_score: float
) -> list[tuple[int, int, float]]:
    """Non-max-suppressed top-N peaks of a score map: ``[(x, y, score), ...]``."""

    min_dist = max(piece_w, piece_h) // 2
    out: list[tuple[int, int, float]] = []
    work = result_map.copy()
    for _ in range(n):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if float(max_val) < min_score:
            break
        x, y = int(max_loc[0]), int(max_loc[1])
        out.append((x, y, float(max_val)))
        x1 = max(0, x - min_dist)
        y1 = max(0, y - min_dist)
        x2 = min(work.shape[1], x + min_dist)
        y2 = min(work.shape[0], y + min_dist)
        work[y1:y2, x1:x2] = 0.0
    return out
