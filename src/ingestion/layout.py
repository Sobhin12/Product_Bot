"""Two-column policy pages that Docling reads as a table.

Legal clauses run in a wide left column, often beside a narrow "Simplified for
you" column. Docling's table model turns such a page into a grid with words from
neighbouring clauses shuffled between cells, so the text is re-read from the PDF
text layer inside the table's bounding box instead.
"""
import re

CLAUSE_CELL = re.compile(r"^\d+(\.\d+)+\.?$")
CLAUSE_HEAD = re.compile(r"^\d+(\.\d+)+\.?\s+\S")
SIDEBAR_X = 0.68  # blocks starting right of this share of the page width may be sidebar
LONG_CELL = 80  # legal clause text is long; a table of short values is not a layout page
MIN_LONG_CELLS = 3


def is_layout_table(grid: list[list[str]]) -> bool:
    """First column holds clause numbers (and blanks for continuation rows) only, and
    the table carries clause-length text. A headerless table of short values (a
    continuation page of "Clause | Benefit | Limit") stays a table."""
    first = [row[0].strip() for row in grid if row]
    numbered = [c for c in first if c]
    if not numbered or not all(CLAUSE_CELL.match(c) for c in numbered):
        return False
    return sum(len(c) > LONG_CELL for row in grid for c in row) >= MIN_LONG_CELLS


def recover_text_blocks(page, bbox_top_left: tuple[float, float, float, float]) -> list[dict]:
    """Text blocks of `page` inside the bbox, in reading order: the left column top
    to bottom, then the right-hand column. Whether the right-hand blocks are a
    sidebar is decided later, for all blocks alike (sidebar.split_sidebars)."""
    l, t, r, b = bbox_top_left
    width, height = page.rect.width, page.rect.height
    found = []
    for x0, y0, x1, y1, text, _, kind in page.get_text("blocks"):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        text = " ".join(text.split())
        if kind != 0 or not text or not (l <= cx <= r and t <= cy <= b):
            continue
        if x0 <= SIDEBAR_X * width and CLAUSE_HEAD.match(text) and len(text) < 120:
            label = "section_header"
        else:
            label = "text"
        found.append(
            {
                "label": label,
                "text": text,
                "bbox": [round(v, 1) for v in (x0, y0, x1, y1)],
                "page_height": height,
                "page_width": width,
            }
        )
    found.sort(key=lambda blk: (blk["bbox"][0] > SIDEBAR_X * width, blk["bbox"][1]))
    return found
