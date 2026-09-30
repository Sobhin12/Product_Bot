"""Stage 2: clause-boundary detection and parent-child chunks for prose blocks.

A clause (4.1) with no sub-clauses is one parent. A clause with sub-clauses gives one
parent per sub-clause (4.1.1, 4.1.2, each with everything under it), prefixed with its
section/clause path, plus one for the clause's own text if it has any."""
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import config
from .sidebar import attach
from .split import count_tokens, split_oversized

NUMERIC = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+\S")  # 4.1 / 4.14.2 (optional trailing dot)
SECTION = re.compile(r"^(\d+)\.\s+\S")  # 4. Benefits
ANNEXURE = re.compile(r"^(?:Annexure|Appendix)\s+(?:[IVXLC]+|\d+)\s*[-–—:.]")  # Annexure V - Co-payments
LETTER = re.compile(r"^([A-Z])\.\s+\S")  # A. / B. ...


@dataclass
class Node:
    header: str
    clause: str | None
    depth: int
    letter: bool = False
    body: list[dict] = field(default_factory=list)
    tables: list[int] = field(default_factory=list)  # real tables sitting in this clause
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None
    section: str | None = None
    page: int = 0  # position of the header block, for pairing sidebars by y
    y: float = 0.0
    sidebars: list[dict] = field(default_factory=list)  # paired sidebar paragraphs


def _line(block: dict) -> str:
    bullet = block["label"] == "list_item" and not block.get("enumerated")
    return f"- {block['text']}" if bullet else block["text"]


def _is_head(block: dict) -> bool:
    """Numbered list items (Docling markers like '4.7.') can be clause headers too."""
    return block["label"] == "section_header" or (
        block["label"] == "list_item" and block.get("enumerated", False)
    )


def build_tree(blocks: list[dict]) -> list[Node]:
    """Return every node in document order. Unnumbered headers stay in the body of
    the clause above, so Notes attach to the sub-clause they follow."""
    root = Node(header="", clause=None, depth=0)
    nodes, stack = [root], [root]
    section = None
    by_heading = not _has_numbered_clauses(blocks)
    for i, b in enumerate(blocks):
        if b["label"] == "table":
            stack[-1].tables.append(b["table_id"])
            continue
        text, head = b["text"], _is_head(b)
        node = None
        if by_heading and _is_section_heading(b):
            level = b.get("level") or 1
            node = Node(text, None, level + 1)  # level 1 is a parent clause, level 2 its child, ...
            if level == 1:
                section = text
        elif head and (m := NUMERIC.match(text)):
            node = Node(text, m.group(1), m.group(1).count(".") + 1)
        elif b["label"] == "section_header" and (m := SECTION.match(text)):
            node = Node(text, m.group(1), 1)
            section = text
        elif b["label"] in ("section_header", "caption", "text") and ANNEXURE.match(text):
            node = Node(text, None, 1)  # an annexure is a section of its own, with no clause number
            section = text
        elif head and _is_letter_header(blocks, i, stack):
            node = Node(text, None, stack[-1].depth + 1, letter=True)
        if node is None:
            stack[-1].body.append(b)
            continue
        if node.letter:
            while stack[-1].letter:
                stack.pop()
        else:
            while stack[-1].depth >= node.depth:
                stack.pop()
        node.parent, node.section = stack[-1], section
        node.page, node.y = b["page"], b["bbox"][1]
        stack[-1].children.append(node)
        stack.append(node)
        nodes.append(node)
    return nodes


def _has_numbered_clauses(blocks: list[dict]) -> bool:
    """False for a document with no "4.1" / "4." style headings at all."""
    return any(
        (_is_head(b) and NUMERIC.match(b["text"]))
        or (b["label"] == "section_header" and SECTION.match(b["text"]))
        for b in blocks
    )


def _is_section_heading(block: dict) -> bool:
    """A Docling heading that starts a section when the document is not numbered. A
    bare label such as "Note:" stays in the body of the section above."""
    if block["label"] not in ("section_header", "title"):
        return False
    text = block["text"].strip()
    return not (text.endswith(":") and len(text.split()) <= 3)


def _is_letter_header(blocks: list[dict], i: int, stack: list[Node]) -> bool:
    """A lettered header counts only inside a numbered clause: 'A.' opens a list (a
    later 'B.' must follow before the next numbered header); any later letter
    continues a list already open in this clause."""
    m = LETTER.match(blocks[i]["text"])
    if not m or stack[-1].depth < 2:
        return False
    letter = m.group(1)
    if letter != "A":
        return stack[-1].letter and letter > LETTER.match(stack[-1].header).group(1)
    for b in blocks[i + 1 :]:
        if not _is_head(b):
            continue
        if NUMERIC.match(b["text"]) or SECTION.match(b["text"]):
            return False
        if (n := LETTER.match(b["text"])) and n.group(1) == "B":
            return True
    return False


def _lines(node: Node, with_descendants: bool) -> list[str]:
    lines = ([node.header] if node.header else []) + [_line(b) for b in node.body]
    if with_descendants:
        for c in node.children:
            lines += _lines(c, True)
    return lines


def _blocks(node: Node, with_descendants: bool) -> list[dict]:
    out = list(node.body)
    if with_descendants:
        for c in node.children:
            out += _blocks(c, True)
    return out


def _tables(node: Node, with_descendants: bool) -> list[int]:
    out = list(node.tables)
    if with_descendants:
        for c in node.children:
            out += _tables(c, True)
    return out


def _sidebars(node: Node, with_descendants: bool) -> list[tuple[Node, dict]]:
    out = [(node, p) for p in node.sidebars]
    if with_descendants:
        for c in node.children:
            out += _sidebars(c, True)
    return out


def _sidebar_groups(pairs: list[tuple[Node, dict]]) -> list[tuple[Node, list[dict]]]:
    """Consecutive paragraphs paired to the same clause on the same page form one chunk."""
    groups: list[tuple[Node, list[dict]]] = []
    for node, para in pairs:
        if groups and groups[-1][0] is node and groups[-1][1][-1]["page"] == para["page"]:
            groups[-1][1].append(para)
        else:
            groups.append((node, [para]))
    return groups


def _has_ancestor_depth2(node: Node) -> bool:
    p = node.parent
    while p:
        if p.depth >= 2:
            return True
        p = p.parent
    return False


def make_chunks(
    file_name: str, blocks: list[dict], sidebars: list[dict], sidebar_log: list[dict]
) -> tuple[list[dict], list[dict]]:
    parents: list[dict] = []
    children: list[dict] = []
    nodes = build_tree(blocks)
    attach(nodes, sidebars, sidebar_log)

    def emit(root: Node, whole_tree: bool, pieces: list[tuple[Node, bool]], path: Sequence[str] = ()):
        """One parent for `root`, its text preceded by `path`; one or more children per
        (node, with_descendants). Children never carry the path."""
        pid = f"P{len(parents) + 1:04d}"
        parents.append(
            _record(
                pid, file_name, root, "\n".join([*path, *_lines(root, whole_tree)]),
                _blocks(root, whole_tree), _tables(root, whole_tree),
            )
        )
        for node, deep in pieces:
            lines = _lines(node, deep)
            text = "\n".join(lines)
            texts = _split(node, lines) if count_tokens(text) > config.MAX_TOKENS else [text]
            for t in texts:
                cid = f"C{len(children) + 1:04d}"
                rec = _record(cid, file_name, node, t, _blocks(node, deep), _tables(node, deep))
                rec["parent_id"] = pid
                rec["split"] = len(texts) > 1
                children.append(rec)
        # Second entry point into the clause: the colloquial sidebar paraphrase.
        for node, paras in _sidebar_groups(_sidebars(root, whole_tree)):
            title = f"{paras[0]['box_title']} – {node.header}"[:100]
            lines = [_line(p) for p in paras]
            text = "\n".join(lines)
            texts = split_oversized(title, lines) if count_tokens(text) > config.MAX_TOKENS else [text]
            for t in texts:
                rec = _record(f"C{len(children) + 1:04d}", file_name, node, t, paras, [])
                rec.update(kind="sidebar", title=title, parent_id=pid, split=len(texts) > 1)
                children.append(rec)

    for node in nodes:
        if node.depth >= 2 and not _has_ancestor_depth2(node):
            if not node.children:
                emit(node, True, [(node, True)])
                continue
            # A clause with sub-clauses: its own text is one parent, and each sub-clause
            # (with everything under it) another, so a match sends the LLM that sub-clause
            # rather than the whole clause. The path keeps the sub-clause readable alone.
            if node.body or node.sidebars:
                emit(node, False, [(node, False)] if node.body else [])
            path = _path(node)
            for c in node.children:
                emit(c, True, [(c, True)], path)
        elif node.depth < 2 and (node.body or node.sidebars or node.tables):
            emit(node, False, [(node, False)] if node.body else [])
    return parents, children


def _path(clause: Node) -> list[str]:
    """Lines above each of `clause`'s sub-clause parents: "<section> > <clause heading>",
    then the clause's own intro text when it is short, since it often decides what the
    sub-clauses mean ("We will not pay for:" makes "5.2.4. Dental treatment" an exclusion)."""
    lines = [" > ".join(dict.fromkeys(h for h in (clause.section, clause.header) if h))]
    intro = [_line(b) for b in clause.body]
    if intro and count_tokens("\n".join(intro)) <= config.INTRO_MAX_TOKENS:
        lines += intro
    return lines


def _clause_of(node: Node) -> str | None:
    """Lettered sub-clauses (A., B., ...) carry the clause number they sit under."""
    while node.letter and node.parent:
        node = node.parent
    return node.clause


def _split(node: Node, lines: list[str]) -> list[str]:
    title = node.header[:100]
    body = lines[1:] if node.header == title else lines
    return split_oversized(title, body)


def _record(rid: str, file_name: str, node: Node, text: str, blocks: list[dict], tables: list[int]) -> dict:
    # The header's own page too: a definition is often all header ("2.1.1. Accident means ...").
    pages = sorted({p for b in blocks for p in b["source_pages"]} | ({node.page} if node.page else set()))
    return {
        "id": rid,
        "file_name": file_name,
        "clauses": [_clause_of(node)] if _clause_of(node) else [],
        "section": node.section,
        "title": node.header[:100],
        "source_pages": pages,
        "tables": tables,
        "contains_placeholder": "<>" in text,
        "tokens": count_tokens(text),
        "text": text,
    }
