"""CIS (Customer Information Sheet) tables: "Sl. No / Title / Description / Clause".

One logical table spread over many pages. Docling's grids for it are inconsistent
from page to page (4, 3, 2 or 1 columns; titles glued to descriptions; clause cells
concatenated), so rows are rebuilt by rule rather than by column position:

- a grid row with a Title cell starts a new logical row (its Sl. No. is inferred
  when the cell is lost); rows without one continue the current row;
- a cell holding only clause numbers is the clause cell of that grid row;
- each grid row is one piece of the logical row (rows are the cut points).
"""
import re
from dataclasses import dataclass, field

from . import config
from .clean import table_grid
from .split import count_tokens, pack

CLAUSE_TOKEN = re.compile(r"^\d+(\.\d+)*(\.[A-Za-z]+)?$")  # 4.1, 5.1.3, 4.1.I, 6.2.3.c
TITLE_CUT = re.compile(r"\s(?=(?:•|[ivx]{1,4}\.|\d{1,2}\.|[A-Za-z]\.|\())")
WIDE = 6  # tables with more columns are premium illustrations, not the CIS


@dataclass
class Piece:
    text: str
    clauses: list[str]
    page: int


@dataclass
class CisRow:
    sl_no: str
    title: str
    pieces: list[Piece] = field(default_factory=list)
    tables: list[int] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


def is_cis_header(grid: list[list[str]]) -> bool:
    row = [c.strip().lower() for c in grid[0]] if grid else []
    return len(row) >= 4 and row[1] == "title" and row[2] == "description"


def cis_range(doc) -> range:
    """Indices of the doc.tables that make up the CIS table (empty if none)."""
    grids = [table_grid(t) for t in doc.tables]
    start = next((i for i, g in enumerate(grids) if is_cis_header(g)), None)
    if start is None:
        return range(0)
    end = next((i for i in range(start + 1, len(grids)) if len(grids[i][0]) > WIDE), len(grids))
    return range(start, end)


def _clause_tokens(cell: str) -> list[str]:
    tokens = [t for t in re.split(r"[\s&,]+", cell) if t]
    return tokens if tokens and all(CLAUSE_TOKEN.match(t) for t in tokens) else []


def _split_title(cell: str) -> tuple[str, str]:
    """Titles are sometimes glued to the start of their description; cut at the
    first list marker."""
    m = TITLE_CUT.search(cell)
    return (cell[: m.start()], cell[m.end() :]) if m else (cell, "")


def build_rows(doc, indices: range) -> list[CisRow]:
    rows: list[CisRow] = []
    last_sl = 0
    for ti in indices:
        table = doc.tables[ti]
        page = table.prov[0].page_no
        grid = table_grid(table)
        for cells in grid[1:] if is_cis_header(grid) else grid:
            cells = [" ".join(c.split()) for c in cells]
            sl, title, desc, clause = _columns(cells)
            tokens = _clause_tokens(clause)
            if clause and not tokens:  # last cell is not a clause cell after all
                desc = f"{desc} {clause}".strip()

            # A title cell holding a lowercase fragment with no Sl. No. is the tail of
            # the previous description, not a new row.
            new_row = len(cells) >= 3 and title and (sl.isdigit() or title[0].isupper())
            if title and not new_row:
                desc = f"{title} {desc}".strip()
            if new_row:
                title, rest = _split_title(title)
                last_sl = int(sl) if sl.isdigit() else last_sl + 1
                rows.append(CisRow(str(last_sl), title))
                desc = f"{rest} {desc}".strip()
            elif not rows:
                rows.append(CisRow("", ""))
            row = rows[-1]
            if ti not in row.tables:
                row.tables.append(ti)
            if page not in row.pages:
                row.pages.append(page)
            if desc:
                row.pieces.append(Piece(desc, tokens, page))
            elif tokens and row.pieces:  # clause number on a row with no text of its own
                row.pieces[-1].clauses += tokens
    return rows


def _columns(cells: list[str]) -> tuple[str, str, str, str]:
    """Return (sl_no, title, description, clause) for a grid row of 1-4+ cells."""
    n = len(cells)
    if n >= 4:
        return cells[0], cells[1], " ".join(cells[2:-1]), cells[-1]
    if n == 3:
        return cells[0], cells[1], cells[2], ""
    if n == 2:
        return "", "", cells[0], cells[1]
    return "", "", cells[0], ""


def make_cis_chunks(
    file_name: str, rows: list[CisRow], parents: list[dict], children: list[dict]
) -> None:
    """Append one parent per CIS row and its children to the given lists."""
    for row in rows:
        lines = [p.text for p in row.pieces]
        text = "\n".join(([row.title] if row.title else []) + lines)
        pid = f"P{len(parents) + 1:04d}"
        parents.append(_record(pid, file_name, row, text, row.pieces, row.pages))
        if count_tokens(text) <= config.MAX_TOKENS:
            texts = [(text, list(range(len(row.pieces))))]
        else:
            texts = pack(row.title, lines)
        for t, origin in texts:
            rec = _record(f"C{len(children) + 1:04d}", file_name, row, t, [row.pieces[i] for i in origin])
            rec["parent_id"] = pid
            rec["split"] = len(texts) > 1
            children.append(rec)


def _record(
    rid: str, file_name: str, row: CisRow, text: str, pieces: list[Piece], pages: list[int] | None = None
) -> dict:
    clauses = list(dict.fromkeys(c for p in pieces for c in p.clauses))
    return {
        "id": rid,
        "file_name": file_name,
        "kind": "cis_row",
        "sl_no": row.sl_no,
        "clauses": clauses,
        "section": "CIS",
        "title": row.title[:100],
        "source_pages": sorted(pages or {p.page for p in pieces} or row.pages),
        "tables": row.tables,
        "contains_placeholder": "<>" in text,
        "tokens": count_tokens(text),
        "text": text,
    }
