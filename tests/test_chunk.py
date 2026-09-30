from ingestion import config
from ingestion.chunk import make_chunks


def block(text: str, label: str = "section_header", page: int = 1, y: float = 100.0, x0: float = 50.0) -> dict:
    return {
        "page": page, "label": label, "enumerated": False, "level": None, "text": text,
        "bbox": [x0, y, x0 + 300, y + 10], "page_height": 800, "page_width": 600, "source_pages": [page],
    }


def text(t: str, y: float = 100.0) -> dict:
    return block(t, label="text", y=y)


EXCLUSIONS = [
    block("5. Exclusions", y=10),
    block("5.2. Specific Exclusions", y=20),
    text("We will not pay for:", y=30),
    block("5.2.1. Dental treatment:", y=40),
    text("All dental treatments other than due to accidents.", y=50),
    block("5.2.2. Cosmetic surgery", y=60),
    text("Cosmetic or aesthetic procedures.", y=70),
]


def chunks(blocks, sidebars=()):
    return make_chunks("f.pdf", blocks, list(sidebars), [])


def by_title(records):
    return {r["title"]: r for r in records}


def test_each_sub_clause_is_its_own_parent_prefixed_with_its_path_and_intro():
    parents, children = chunks(EXCLUSIONS)
    p = by_title(parents)
    assert set(p) == {"5.2. Specific Exclusions", "5.2.1. Dental treatment:", "5.2.2. Cosmetic surgery"}

    dental = p["5.2.1. Dental treatment:"]
    assert dental["text"] == (
        "5. Exclusions > 5.2. Specific Exclusions\n"
        "We will not pay for:\n"
        "5.2.1. Dental treatment:\n"
        "All dental treatments other than due to accidents."
    )
    assert dental["clauses"] == ["5.2.1"] and dental["section"] == "5. Exclusions"
    # the clause's own intro is also a parent of its own, so it stays searchable
    assert p["5.2. Specific Exclusions"]["text"] == "5.2. Specific Exclusions\nWe will not pay for:"


def test_children_are_the_bare_sub_clause_and_point_at_their_own_parent():
    parents, children = chunks(EXCLUSIONS)
    pid = by_title(parents)["5.2.1. Dental treatment:"]["id"]
    kids = [c for c in children if c["parent_id"] == pid]
    assert [c["text"] for c in kids] == ["5.2.1. Dental treatment:\nAll dental treatments other than due to accidents."]


def test_a_long_intro_stays_out_of_the_path():
    long_intro = "Subject to the waiting periods and limits in the Schedule, " * 6  # well over 60 tokens
    assert len(long_intro) // 4 > config.INTRO_MAX_TOKENS
    blocks = [*EXCLUSIONS[:2], text(long_intro, y=30), *EXCLUSIONS[3:]]
    parents, _ = chunks(blocks)
    p = by_title(parents)
    assert p["5.2.1. Dental treatment:"]["text"].startswith("5. Exclusions > 5.2. Specific Exclusions\n5.2.1. Dental")
    assert long_intro in p["5.2. Specific Exclusions"]["text"]


def test_a_clause_without_sub_clauses_is_unchanged():
    parents, children = chunks([block("4. Benefits", y=10), block("4.44. Critical Illness", y=20), text("Pays the sum insured.", y=30)])
    assert [p["text"] for p in parents] == ["4.44. Critical Illness\nPays the sum insured."]
    assert [c["text"] for c in children] == ["4.44. Critical Illness\nPays the sum insured."]


def test_a_clause_with_no_intro_gets_no_parent_of_its_own():
    blocks = [b for b in EXCLUSIONS if b["text"] != "We will not pay for:"]
    parents, _ = chunks(blocks)
    assert "5.2. Specific Exclusions" not in by_title(parents)
    assert by_title(parents)["5.2.2. Cosmetic surgery"]["text"].startswith(
        "5. Exclusions > 5.2. Specific Exclusions\n5.2.2. Cosmetic surgery"
    )


def test_a_sub_clause_keeps_everything_nested_under_it_in_one_parent():
    blocks = [
        *EXCLUSIONS[:4],
        block("5.2.1.1. Implants", y=42), text("Dental implants.", y=44),
        block("5.2.1.2. Braces", y=46), text("Orthodontic braces.", y=48),
        *EXCLUSIONS[5:],
    ]
    parents, _ = chunks(blocks)
    p = by_title(parents)
    assert "5.2.1.1. Implants" not in p
    assert "Dental implants." in p["5.2.1. Dental treatment:"]["text"]
    assert "Orthodontic braces." in p["5.2.1. Dental treatment:"]["text"]


def test_an_oversized_sub_clause_is_one_whole_parent_with_split_children():
    long_body = [text(f"Sentence number {i} about the dental exclusion and its many conditions.", y=40 + i * 0.01)
                 for i in range(60)]  # ~60 * 18 tokens, well over MAX_TOKENS
    blocks = [*EXCLUSIONS[:4], *long_body, *EXCLUSIONS[5:]]
    parents, children = chunks(blocks)
    dental = by_title(parents)["5.2.1. Dental treatment:"]
    kids = [c for c in children if c["parent_id"] == dental["id"]]
    assert len(kids) > 1 and all(c["split"] for c in kids)
    assert all(b["text"] in dental["text"] for b in long_body)  # nothing cut from the parent


def test_a_sub_clause_that_is_all_header_still_cites_its_page():
    parents, _ = chunks([
        block("2. Definitions", page=4, y=10), block("2.1. Standard Definitions:", page=4, y=20),
        block("2.1.1. Accident means a sudden, unforeseen event.", page=5, y=30),
        block("2.1.2. Hospital means an institution with beds.", page=5, y=40),
    ])
    assert by_title(parents)["2.1.1. Accident means a sudden, unforeseen event."]["source_pages"] == [5]


def test_a_sidebar_goes_with_the_sub_clause_it_is_printed_beside():
    sidebar = {**text("Cosmetic work is not covered unless it follows an accident.", y=61), "source_pages": [1]}
    parents, children = chunks(EXCLUSIONS, sidebars=[sidebar])
    cosmetic = by_title(parents)["5.2.2. Cosmetic surgery"]
    side = [c for c in children if c.get("kind") == "sidebar"]
    assert len(side) == 1 and side[0]["parent_id"] == cosmetic["id"]
