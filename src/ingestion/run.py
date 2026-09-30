"""Stages 1-2: parse PDFs in data/input, clean blocks, dump tables, chunk prose
into parents/children in data/output."""
import json
import sys
from collections.abc import Callable
from pathlib import Path

from . import config
from .chunk import make_chunks
from .cis import build_rows, cis_range, make_cis_chunks
from .clean import extract_blocks, stitch_pages, strip_boilerplate, table_grid
from .parse import parse_pdf
from .sidebar import split_sidebars
from .tables import build_chains, make_table_chunks, table_kinds


TABLE_KINDS = ("table", "lookup_table", "list_table", "entry_table")


def table_headings(blocks: list[dict]) -> dict[int, list[str]]:
    """Headings and captions between each table and the one before it; a table's own
    caption (Docling links it, though it follows the table in reading order) comes
    last, so it is the table's heading, and is not a heading of the next table."""
    out: dict[int, list[str]] = {}
    pending: list[str] = []
    own_caption = ""
    for b in blocks:
        if b["label"] == "caption" and b["text"] == own_caption:
            continue
        if b["label"] in ("section_header", "caption"):
            pending.append(b["text"])
        elif b["label"] == "table":
            own_caption = b["caption"]
            out[b["table_id"]], pending = pending + ([own_caption] if own_caption else []), []
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def dump_tables(doc, out: Path, kinds: list[str]) -> int:
    tables_dir = out / "tables"
    tables_dir.mkdir(exist_ok=True)
    for i, table in enumerate(doc.tables):
        grid = table_grid(table)
        payload = {
            "kind": kinds[i],
            "page": table.prov[0].page_no,
            "num_rows": table.data.num_rows,
            "num_cols": table.data.num_cols,
            "grid": grid,
        }
        (tables_dir / f"table_{i:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (tables_dir / f"table_{i:03d}.md").write_text(
            table.export_to_markdown(doc=doc), encoding="utf-8"
        )
    return len(doc.tables)


def write_report(path: Path, parents: list[dict], children: list[dict]) -> None:
    by_parent: dict[str, list[dict]] = {}
    for c in children:
        by_parent.setdefault(c["parent_id"], []).append(c)
    lines = []
    for p in parents:
        kids = by_parent.get(p["id"], [])
        label = (
            f"CIS {p['sl_no']}: {p['title']}"
            if p.get("kind") == "cis_row"
            else p["title"]
            if p.get("kind") in TABLE_KINDS
            else ", ".join(p["clauses"]) or "front matter"
        )
        lines.append(
            f"## {p['id']} · {label} · p{p['source_pages']} · ~{p['tokens']} tok · {len(kids)} child(ren)\n"
        )
        for c in kids:
            flag = " · SPLIT" if c["split"] else ""
            lines.append(f"**{c['id']}** ~{c['tokens']} tok{flag}\n\n```\n{c['text']}\n```\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def process(pdf: Path, on_stage: Callable[[str], None] = lambda stage: None) -> tuple[list[dict], list[dict]]:
    """Parse and chunk one PDF into data/output/<stem>; returns (parents, children)."""
    print(f"[{pdf.name}] parsing...", flush=True)
    on_stage("parsing")
    doc = parse_pdf(pdf)
    on_stage("chunking")

    out = config.OUTPUT_DIR / pdf.stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "docling.md").write_text(doc.export_to_markdown(), encoding="utf-8")

    blocks = extract_blocks(doc, pdf)
    kept, removed = strip_boilerplate(blocks)
    main, sidebars, warnings = split_sidebars(kept)
    for w in warnings:
        print(f"[{pdf.name}] WARNING {w}")
    stitched = stitch_pages(main)
    write_jsonl(out / "blocks.jsonl", stitched)
    write_jsonl(out / "removed_boilerplate.jsonl", removed)
    kinds = table_kinds(doc)
    n_tables = dump_tables(doc, out, kinds)

    sidebar_log: list[dict] = []
    parents, children = make_chunks(pdf.name, stitched, sidebars, sidebar_log)
    make_cis_chunks(pdf.name, build_rows(doc, cis_range(doc)), parents, children)
    make_table_chunks(pdf.name, build_chains(doc, kinds, table_headings(stitched)), parents, children)
    write_jsonl(out / "sidebars.jsonl", sidebar_log)
    write_jsonl(out / "parents.jsonl", parents)
    write_jsonl(out / "children.jsonl", children)
    write_report(out / "report.md", parents, children)

    print(
        f"[{pdf.name}] {len(blocks)} blocks -> {len(removed)} boilerplate removed, "
        f"{len(main) - len(stitched)} stitched, {len(sidebars)} sidebar blocks, {len(stitched)} kept; {n_tables} tables; "
        f"{len(parents)} parents, {len(children)} children"
    )
    return parents, children


def main(argv: list[str]) -> None:
    pdfs = [Path(a) for a in argv] or sorted(config.INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {config.INPUT_DIR}")
    for pdf in pdfs:
        process(pdf)


if __name__ == "__main__":
    main(sys.argv[1:])
