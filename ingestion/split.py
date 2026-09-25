import re

from . import config

# Not after a bare list number ("20."), which belongs to the item that follows.
SENTENCE_END = re.compile(r"(?<!\s\d\.)(?<!\s\d\d\.)(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    """Rough estimate (~4 chars/token). Swap for the embedding model's tokenizer."""
    return len(text) // 4 + 1


def split_oversized(header: str, lines: list[str], max_tokens: int = config.MAX_TOKENS) -> list[str]:
    """Pack `lines` (blocks/bullets/rows are the cut points) into pieces of at most
    max_tokens, each prefixed with `header`. A single line over the limit is split
    at sentence boundaries, then at word boundaries."""
    return [text for text, _ in pack(header, lines, max_tokens)]


def pack(header: str, lines: list[str], max_tokens: int = config.MAX_TOKENS) -> list[tuple[str, list[int]]]:
    """Like split_oversized, but each piece also lists the indices of the source
    lines it draws from."""
    prefix = f"{header}\n" if header else ""
    budget = max(max_tokens - count_tokens(prefix), 1)

    pieces: list[tuple[str, list[int]]] = []
    current: list[str] = []
    origin: list[int] = []
    used = 0
    for i, source in enumerate(lines):
        for line in _fit(source, budget):
            cost = count_tokens(line) + 1
            if current and used + cost > budget:
                pieces.append((prefix + "\n".join(current), origin))
                current, origin, used = [], [], 0
            current.append(line)
            if not origin or origin[-1] != i:
                origin.append(i)
            used += cost
    if current:
        pieces.append((prefix + "\n".join(current), origin))
    return pieces


def _fit(line: str, budget: int) -> list[str]:
    if count_tokens(line) <= budget:
        return [line]
    out: list[str] = []
    for sentence in SENTENCE_END.split(line):
        if count_tokens(sentence) <= budget:
            out.append(sentence)
            continue
        words, part = sentence.split(), []
        for w in words:
            if part and count_tokens(" ".join(part + [w])) > budget:
                out.append(" ".join(part))
                part = []
            part.append(w)
        if part:
            out.append(" ".join(part))
    return out
