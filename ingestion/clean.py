import re
from collections import defaultdict
from pathlib import Path

import pymupdf
from docling_core.types.doc import DoclingDocument, TableItem

from . import config
from .layout import is_layout_table, recover_text_blocks

TERMINAL = (".", "!", "?", ":", ";", '"', "”", ")")
PROSE = ("text", "list_item")


def _norm(text: str) -> str:
    """Collapse whitespace and mask digits so 'Page 3 of 13' == 'Page 4 of 13'."""
    return re.sub(r"\d+", "#", " ".join(text.split())).lower()


def table_grid(table: TableItem) -> list[list[str]]:
    return [[cell.text for cell in row] for row in table.data.grid]


def extract_blocks(doc: DoclingDocument, pdf: Path) -> list[dict]:
    """Flat, reading-ordered list of blocks with page and bbox. A real table becomes
    a `table` marker block (its `table_id` indexes doc.tables) so it keeps its place
    in the clause flow. A two-column layout that Docling misread as a table is
    replaced by text re-read from the PDF (see layout.py)."""
    blocks = []
    table_ids = {t.self_ref: i for i, t in enumerate(doc.tables)}
    pdf_doc = pymupdf.open(pdf)
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem):
            blocks += _table_blocks(doc, pdf_doc, item, table_ids[item.self_ref])
            continue
        if not getattr(item, "text", "").strip():
            continue
        prov = item.prov[0]
        height = doc.pages[prov.page_no].size.height
        bbox = prov.bbox.to_top_left_origin(height)
        # Docling keeps list numbering ("4.7.", "A.") in `marker`, apart from the text.
        # Bullet glyphs are not enumerated and are dropped.
        enumerated = bool(getattr(item, "enumerated", False))
        text = item.text.strip()
        if enumerated and getattr(item, "marker", None):
            text = f"{item.marker.strip()} {text}"
        blocks.append(
            {
                "page": prov.page_no,
                "label": str(item.label.value),
                "enumerated": enumerated,
                "level": getattr(item, "level", None),  # heading level, when Docling has one
                "text": text,
                "bbox": [round(v, 1) for v in (bbox.l, bbox.t, bbox.r, bbox.b)],
                "page_height": height,
                "page_width": doc.pages[prov.page_no].size.width,
            }
        )
    return blocks


def _table_blocks(doc: DoclingDocument, pdf_doc, table: TableItem, table_id: int) -> list[dict]:
    prov = table.prov[0]
    height = doc.pages[prov.page_no].size.height
    bbox = prov.bbox.to_top_left_origin(height)
    box = (bbox.l, bbox.t, bbox.r, bbox.b)
    if not is_layout_table(table_grid(table)):
        return [
            {
                "page": prov.page_no,
                "label": "table",
                "table_id": table_id,
                "caption": table.caption_text(doc),
                "text": "",
                "bbox": [round(v, 1) for v in box],
                "page_height": height,
            }
        ]
    recovered = recover_text_blocks(pdf_doc[prov.page_no - 1], box)
    return [{"page": prov.page_no, "enumerated": False, "recovered_from_table": table_id, **b} for b in recovered]


def strip_boilerplate(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop repeated header/footer text. Returns (kept, removed)."""
    pages = {b["page"] for b in blocks}
    seen = defaultdict(set)  # normalised text -> pages where it sits in a band
    for b in blocks:
        if b["label"] != "table" and _in_band(b):
            seen[_norm(b["text"])].add(b["page"])
    threshold = max(config.MIN_REPEAT_PAGES, config.REPEAT_SHARE * len(pages))
    boilerplate = {t for t, p in seen.items() if len(p) >= threshold}

    kept, removed = [], []
    for b in blocks:
        drop = b["label"] != "table" and _in_band(b) and _norm(b["text"]) in boilerplate
        (removed if drop else kept).append(b)
    return kept, removed


def _in_band(b: dict) -> bool:
    top, bottom, height = b["bbox"][1], b["bbox"][3], b["page_height"]
    band = config.BAND_FRACTION * height
    return top < band or bottom > height - band


def stitch_pages(blocks: list[dict]) -> list[dict]:
    """Join a block ending mid-sentence on page N with a lowercase-starting
    block on page N+1. Adds `source_pages` to every block."""
    out: list[dict] = []
    for b in blocks:
        b = {**b, "source_pages": [b["page"]]}
        prev = out[-1] if out else None
        if (
            prev
            and b["page"] == prev["source_pages"][-1] + 1
            and prev["label"] in PROSE and b["label"] in PROSE
            and not prev["text"].endswith(TERMINAL)
            and b["text"][:1].islower()
        ):
            prev["text"] += " " + b["text"]
            prev["source_pages"].append(b["page"])
            continue
        out.append(b)
    return out
