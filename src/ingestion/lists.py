"""Flat reference lists: Annexure I non-payables (List I-IV), specified illnesses,
vaccinations, tests covered, ombudsman offices.

- Numbered lists ("Sl. No. | Item" repeated side by side, or a single pair) are
  unpivoted so each list reads in order: "1. BABY FOOD" ... "68. VASOFIX SAFETY".
- A row of identical headers ("List of tests covered:" x3) is one list of all cells.
- Two-column entry tables (Office Details | Jurisdiction) give one entry per row.
"""
import re

SL_HEADER = re.compile(r"^sl?\.?\s*no\.?$", re.I)  # S.No / S. No. / Sl. No.
GENERIC_ITEM = {"item", "items"}
MAX_LIST_COLS = 4  # wider grids with a merged title (premium illustrations) are not lists
SHORT_LABEL = 3  # a lone city name of at most this many words is glued to the next row


def _clean(cell: str) -> str:
    return " ".join(cell.split())


def numbered_list(header: list[str], rows: list[list[str]]) -> tuple[list[str], str] | None:
    """Return (lines, item header) for "Sl. No. | Item" pairs, or None."""
    n = len(header)
    if n < 2 or n % 2 or not all(SL_HEADER.match(_clean(header[k])) for k in range(0, n, 2)):
        return None
    if not all(_clean(header[k]) and not SL_HEADER.match(_clean(header[k])) for k in range(1, n, 2)):
        return None
    lines = []
    for k in range(0, n, 2):  # one list per pair of columns, read top to bottom
        for r in rows:
            sl, item = _clean(r[k]), _clean(r[k + 1])
            if item:
                lines.append(f"{sl}. {item}" if sl else item)
    return (lines, _clean(header[1])) if lines else None


def repeated_header_list(header: list[str], rows: list[list[str]]) -> tuple[list[str], str] | None:
    """Every header cell identical: the columns are one list laid out in a grid."""
    heads = {_clean(h) for h in header}
    if not 2 <= len(header) <= MAX_LIST_COLS or len(heads) != 1 or not next(iter(heads)):
        return None
    lines = [_clean(c) for r in rows for c in r if _clean(c)]
    return (lines, next(iter(heads))) if lines else None


def item_title(base: str, item_header: str) -> str:
    """List title, adding the column header when it says more than "Item"."""
    label = item_header.rstrip(":")
    return base if label.lower() in GENERIC_ITEM else f"{base} – {label}"


def entry_rows(header: list[str], rows: list[list[str]], title: str) -> list[tuple[str, list[str]]] | None:
    """Two-column tables of entries, e.g. (office details, jurisdiction): one sentence
    per row. A row holding only a short label (a city name split from its entry by
    the table model) is joined to the row below it."""
    if len(header) != 2 or not all(_clean(h) for h in header) or len(rows) < 3:
        return None
    merged: list[list[str]] = []
    carry = ""
    for r in rows:
        first, second = _clean(r[0]), _clean(r[1])
        if not second and 0 < len(first.split()) <= SHORT_LABEL:
            carry = f"{carry} {first}".strip()
            continue
        merged.append([f"{carry} {first}".strip(), second])
        carry = ""
    if carry:
        merged.append([carry, ""])
    h0, h1 = _clean(header[0]), _clean(header[1])
    out = []
    for first, second in merged:
        parts = [f"{h0}: {first}"] + ([f"{h1}: {second}"] if second else [])
        out.append((f"{title}. " + ". ".join(parts) + ".", [first, second]))
    return out
