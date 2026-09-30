"""Lookup tables (condition -> value): PTD/PPD percentages, room-category
co-payment, discount-by-points.

A table is a lookup table when its trailing column(s) are mostly percentages. Each
data row becomes one child whose text is a sentence built from a fixed template
(no LLM): "<clause title>. <condition header>: <condition>. <value header>: <value>."
Multi-tier headers are flattened into one string per column, and columns that are
blank below their first cell (vertically merged) are filled downwards.
"""
import re

PERCENT = re.compile(r"^\s*(?:up to |>=?|<=?)?\s*\d+(?:\.\d+)?\s*%\s*$", re.I)
BULLET = re.compile(r"\s*[•●▪�]\s*")
HEADER_MAX = 80  # header cells sometimes carry footnotes; keep the label part only
VALUE_SHARE = 0.8  # share of a column's cells that must be percentages
MIN_DATA_ROWS = 2


def _is_value(cell: str) -> bool:
    return bool(PERCENT.match(cell))


def _clean(cell: str) -> str:
    cell = " ".join(cell.split())
    return BULLET.sub("; ", cell).strip("; ").replace(":; ", ": ")


def _label(cell: str) -> str:
    """Header text without a trailing "NOTE: ..." footnote, capped at HEADER_MAX chars."""
    text = re.split(r"\s+NOTE\b", _clean(cell))[0]
    if len(text) <= HEADER_MAX:
        return text
    return text[:HEADER_MAX].rsplit(" ", 1)[0] + "…"


def lookup_rows(
    header: list[str], rows: list[list[str]], context: str
) -> tuple[list[tuple[str, list[str]]], bool] | None:
    """Return ((sentence, source row) per data row, needs_review), or None if this is
    not a lookup table. A percentage inside a column name means Docling folded a data
    row into the header, so the table is flagged for review."""
    grid = ([header] if header else []) + rows
    n = len(grid[0])
    if n < 2 or len(grid) < MIN_DATA_ROWS + 1:
        return None

    def value_column(c: int) -> bool:
        cells = [r[c] for r in grid[1:] if r[c].strip()]
        return len(cells) >= MIN_DATA_ROWS and sum(map(_is_value, cells)) / len(cells) >= VALUE_SHARE

    value_cols: list[int] = []
    c = n - 1
    while c >= 1 and value_column(c):
        value_cols.insert(0, c)
        c -= 1
    if not value_cols:
        return None
    cond_cols = list(range(value_cols[0]))

    h = 0  # leading rows with no percentage are header rows (multi-tier headers)
    while h < len(grid) and not any(_is_value(grid[h][c]) for c in value_cols):
        h += 1
    header_rows, data = grid[:h], [list(r) for r in grid[h:]]
    if len(data) < MIN_DATA_ROWS:
        return None

    names = [
        " – ".join(dict.fromkeys(_label(r[c]) for r in header_rows if r[c].strip())) or f"Column {c + 1}"
        for c in range(n)
    ]
    for c in cond_cols:  # a mostly-blank column is a vertical merge: fill it downwards
        filled = [r[c] for r in data if r[c].strip()]
        if filled and len(filled) <= len(data) / 3:
            last = ""
            for r in data:
                last = r[c] if r[c].strip() else last
                r[c] = last

    context = context.rstrip(".: ")
    out = []
    for raw, r in zip(grid[h:], data):
        parts = [f"{names[c]}: {_clean(r[c])}" for c in cond_cols + value_cols if r[c].strip()]
        if any(r[c].strip() for c in value_cols):
            out.append((f"{context}. " + ". ".join(parts) + ".", raw))
    needs_review = any(re.search(r"\d\s*%", n) for n in names)
    return (out, needs_review) if out else None
