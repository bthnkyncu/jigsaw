"""Window-title matching is case-insensitive.

The dev table reads "Masa #2 @ YapBoz Salonu 2"; a customer's window may read
"Yapboz oyun salonu" (lower-case 'b'). The same default substring must catch
both, so matching ignores case.
"""

from __future__ import annotations

from puzzle_assistant.capture.interfaces import title_matches


def test_matches_dev_table_title() -> None:
    assert title_matches("Masa #2 @ YapBoz Salonu 2 [Yönetici: x]", "YapBoz")


def test_matches_customer_lowercase_title() -> None:
    # Customer title differs in case; default substring must still match.
    assert title_matches("Yapboz oyun salonu", "YapBoz")
    assert title_matches("Yapboz oyun salonu", "yapboz")


def test_rejects_unrelated_title() -> None:
    assert not title_matches("Mozilla Firefox", "YapBoz")
