"""Sidebar pairing (LLD Step 5): the "Simplified for you" / "What it means?" column
printed beside legal clauses.

Docling emits a sidebar after the left-column text, so left alone each paragraph
lands in whichever clause ends the page. Instead, sidebar blocks are taken out of
the text flow and each paragraph is paired with the clause whose header is the last
one at or above it in the page (a paragraph at the top of a page therefore goes to
the clause carried over from the previous page). A paragraph whose text was already
paired earlier is a repeat and is skipped.
"""
from .layout import SIDEBAR_X

TITLE_WORDS = 4  # a short header in the sidebar column is the box title, not content
Y_TOLERANCE = 3.0  # points; a paragraph level with a clause header belongs to it

# A right-hand column is a sidebar only if it looks like one: narrow, near the right
# margin, made of prose, and a small part of the page. Measured on the sample files,
# real sidebars are 0.10-0.25 wide, end at 0.80-0.95, average 58+ characters per
# block and hold 3-32% of the page text.
MAX_WIDTH = 0.30  # block width / page width
RIGHT_EDGE_MIN = 0.75  # block right edge / page width
MIN_MEAN_CHARS = 50  # table cells ("INR 2,000") are much shorter than prose
MAX_PAGE_SHARE = 0.40  # above this it is a two-column layout, not a sidebar


def _in_band(block: dict) -> bool:
    """Geometry only: starts right of SIDEBAR_X, is narrow, ends near the right margin."""
    if block["label"] == "table":
        return False
    x0, _, x1, _ = block["bbox"]
    w = block["page_width"]
    return x0 > SIDEBAR_X * w and (x1 - x0) <= MAX_WIDTH * w and x1 >= RIGHT_EDGE_MIN * w


def split_sidebars(blocks: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Return (main-column blocks, sidebar blocks, warnings). The blocks in the right-hand
    band of a page are a sidebar only if they pass the prose and page-share tests;
    otherwise they stay in the main flow and a warning says why."""
    by_page: dict[int, list[dict]] = {}
    for b in blocks:
        if _in_band(b):
            by_page.setdefault(b["page"], []).append(b)
    page_chars: dict[int, int] = {}
    for b in blocks:
        if b["label"] != "table":
            page_chars[b["page"]] = page_chars.get(b["page"], 0) + len(b["text"])

    accepted: set[int] = set()
    warnings: list[str] = []
    for page, band in by_page.items():
        chars = sum(len(b["text"]) for b in band)
        mean, share = chars / len(band), chars / page_chars[page]
        if mean < MIN_MEAN_CHARS:
            warnings.append(f"p{page}: {len(band)} right-hand blocks kept in the text (mean {mean:.0f} chars, too short for prose - table column?)")
        elif share > MAX_PAGE_SHARE:
            warnings.append(f"p{page}: {len(band)} right-hand blocks kept in the text ({share:.0%} of the page - two-column layout?)")
        else:
            accepted.update(id(b) for b in band)

    main, side = [], []
    for b in blocks:
        if id(b) in accepted:
            side.append({**b, "source_pages": [b["page"]]})
        else:
            main.append(b)
    return main, side, warnings


def _node_at(nodes: list, page: int, y: float):
    for node in reversed(nodes):
        if (node.page, node.y) <= (page, y + Y_TOLERANCE):
            return node
    return nodes[0]


def attach(nodes: list, sidebars: list[dict], log: list[dict]) -> None:
    """Append each sidebar paragraph to `node.sidebars` of its clause node; record
    what happened to every sidebar block in `log`."""
    seen: dict[str, str] = {}
    titles: dict[int, str] = {}
    for b in sorted(sidebars, key=lambda b: (b["page"], b["bbox"][1])):
        node = _node_at(nodes, b["page"], b["bbox"][1])
        entry = {
            "page": b["page"],
            "y": b["bbox"][1],
            "clause": node.header[:60],
            "text": b["text"],
        }
        key = " ".join(b["text"].lower().split())
        if b["label"] == "section_header" and len(b["text"].split()) <= TITLE_WORDS:
            titles[b["page"]] = b["text"]
            entry["status"] = "title"
        elif key in seen:
            entry.update(status="duplicate", duplicate_of=seen[key])
        else:
            seen[key] = node.header[:60]
            node.sidebars.append({**b, "box_title": titles.get(b["page"], "Sidebar")})
            entry["status"] = "paired"
        log.append(entry)
