"""Tables that are not CIS or layout tables: lookup tables, flat lists, entry tables,
and a generic fallback for the rest.

A table that fits in max_tokens is one chunk (child and parent). A larger one is a
parent (the whole table) with children packed from its rows, the header repeated on
each. A table that starts at the top of the page after another table with the same
column count continues it and inherits its header.
"""
from dataclasses import dataclass, field

from . import config
from .clean import table_grid
from .cis import cis_range
from .layout import is_layout_table
from .lists import entry_rows, item_title, numbered_list, repeated_header_list
from .lookup import lookup_rows
from .split import count_tokens, pack

TOP_BAND = 0.20  # a continuation starts in the top 20% of its page ...
BOTTOM_BAND = 0.65  # ... and the table before it ends in the bottom 35% of its page


@dataclass
class Chain:
    header: list[str]
    rows: list[list[str]]
    pages: list[int]
    ids: list[int]
    cols: int
    bottom: float = 0.0  # bottom edge of the last table, as a share of page height
    height: float = 0.0
    heading: str = ""  # latest heading / caption seen just before one of its tables


def table_kinds(doc) -> list[str]:
    cis = cis_range(doc)
    return [
        "cis" if i in cis else "layout" if is_layout_table(table_grid(t)) else "table"
        for i, t in enumerate(doc.tables)
    ]


def build_chains(doc, kinds: list[str], headings: dict[int, list[str]]) -> list[Chain]:
    """`headings[i]` are the headings/captions between table i-1 and table i in reading
    order. A table that repeats the previous header after a heading is a new table
    (List II, then List III); a continuation without a header stays joined even when a
    heading was captured between them (Docling puts a caption after its table)."""
    chains: list[Chain] = []
    for i, table in enumerate(doc.tables):
        grid = table_grid(table)
        if kinds[i] != "table" or not any(c.strip() for row in grid for c in row):
            continue
        prov = table.prov[0]
        height = doc.pages[prov.page_no].size.height
        box = prov.bbox.to_top_left_origin(height)
        heading = headings.get(i, [""])[-1] if headings.get(i) else ""
        prev = chains[-1] if chains else None
        if (
            prev
            and prev.ids[-1] == i - 1
            and prov.page_no == prev.pages[-1] + 1
            and len(grid[0]) == prev.cols
            and not _new_header(grid[0], prev.header)
            and not (grid[0] == prev.header and heading)
            and box.t < TOP_BAND * height
            and prev.bottom > BOTTOM_BAND
        ):
            rows = grid[1:] if grid[0] == prev.header else grid
            prev.rows += rows
            prev.pages.append(prov.page_no)
            prev.ids.append(i)
            prev.bottom = box.b / height
            prev.heading = heading or prev.heading
            continue
        header, rows = (grid[0], grid[1:]) if len(grid) > 1 else ([], grid)
        chains.append(Chain(header, rows, [prov.page_no], [i], len(grid[0]), box.b / height, height, heading))
    return chains


def _new_header(first_row: list[str], header: list[str]) -> bool:
    """A first row that starts like the previous header but is not identical to it is
    the header of a different table ("Policy Start Date | End of 21 months | ...",
    "Condition for Permanent Total / Partial Disability")."""
    if not header or first_row == header:
        return False
    return first_row[0].lower().split()[:1] == header[0].lower().split()[:1] != []


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(" ".join(c.split()).replace("|", "/") for c in cells) + " |"


def make_table_chunks(
    file_name: str, chains: list[Chain], parents: list[dict], children: list[dict]
) -> None:
    """Append a parent and children per table chain. Clause context comes from the
    prose child that holds the table's marker, so call this after the prose chunks."""
    owner: dict[int, dict] = {}
    for c in children:
        for tid in c.get("tables", []):
            owner.setdefault(tid, c)
    for p in parents:  # a section with only a table under it has no child of its own
        for tid in p.get("tables", []):
            owner.setdefault(tid, p)

    for chain in chains:
        ctx = owner.get(chain.ids[0], {})
        pages = chain.pages if len(chain.pages) == 1 else [chain.pages[0], chain.pages[-1]]
        where = f"pages {pages[0]}-{pages[-1]}" if len(pages) > 1 else f"page {pages[0]}"
        title = f"Table under {ctx['title']} ({where})" if ctx.get("title") else f"Table ({where})"
        header = [_md_row(chain.header), _md_row(["---"] * chain.cols)] if chain.header else []
        rows = [_md_row(r) for r in chain.rows]
        text = "\n".join([title, *header, *rows])

        lookup = lookup_rows(chain.header, chain.rows, ctx.get("title") or title)
        rows_as_sentences = lookup[0] if lookup else None
        pid = f"P{len(parents) + 1:04d}"
        parent = _record(pid, file_name, chain, ctx, title, text)
        if rows_as_sentences:
            parent.update(kind="lookup_table", needs_review=lookup[1])
        parents.append(parent)
        if rows_as_sentences:
            for sentence, raw in rows_as_sentences:
                rec = _record(f"C{len(children) + 1:04d}", file_name, chain, ctx, title, sentence)
                rec.update(
                    kind="lookup_row", parent_id=pid, split=False,
                    source_row=_md_row(raw), needs_review=lookup[1],
                )
                children.append(rec)
            continue

        flat = numbered_list(chain.header, chain.rows) or repeated_header_list(chain.header, chain.rows)
        if flat:
            lines, item_header = flat
            list_title = item_title(chain.heading or ctx.get("title") or title, item_header)
            body = "\n".join([list_title, *lines])
            parent.update(kind="list_table", title=list_title[:100], text=body, tokens=count_tokens(body))
            pieces = [body] if count_tokens(body) <= config.MAX_TOKENS else [t for t, _ in pack(list_title, lines)]
            for t in pieces:
                rec = _record(f"C{len(children) + 1:04d}", file_name, chain, ctx, list_title, t)
                rec.update(kind="list", parent_id=pid, split=len(pieces) > 1)
                children.append(rec)
            continue

        entry_title = chain.heading or ctx.get("title") or title
        entries = entry_rows(chain.header, chain.rows, entry_title)
        if entries:
            parent.update(kind="entry_table", title=entry_title[:100])
            for sentence, raw in entries:
                rec = _record(f"C{len(children) + 1:04d}", file_name, chain, ctx, entry_title, sentence)
                rec.update(kind="entry_row", parent_id=pid, split=False, source_row=_md_row(raw))
                children.append(rec)
            continue

        if count_tokens(text) <= config.MAX_TOKENS:
            pieces = [text]
        else:
            prefix = "\n".join([title, *header])
            if count_tokens(prefix) > config.MAX_TOKENS // 2:  # very wide header: keep the title only
                prefix = title
            pieces = [t for t, _ in pack(prefix, rows)]
        for t in pieces:
            rec = _record(f"C{len(children) + 1:04d}", file_name, chain, ctx, title, t)
            rec["parent_id"] = pid
            rec["split"] = len(pieces) > 1
            children.append(rec)


def _record(rid: str, file_name: str, chain: Chain, ctx: dict, title: str, text: str) -> dict:
    return {
        "id": rid,
        "file_name": file_name,
        "kind": "table",
        "clauses": ctx.get("clauses", []),
        "section": ctx.get("section"),
        "title": title[:100],
        "source_pages": chain.pages,
        "tables": chain.ids,
        "contains_placeholder": "<>" in text,
        "tokens": count_tokens(text),
        "text": text,
    }
