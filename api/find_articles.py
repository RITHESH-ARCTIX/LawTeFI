"""Find which chunks (and labels) actually contain the text of specific Constitution articles.

Put this file in D:\\LawTeFI\\api, activate the venv, then run:
    python find_articles.py
It only reads the index; nothing is changed. Paste the output back to Claude.
"""

from __future__ import annotations

from app.retrieval import _collection

records = _collection().get(include=["documents", "metadatas"])
chunks = [
    (m or {}, d or "")
    for m, d in zip(records["metadatas"], records["documents"])
    if "constitution" in str((m or {}).get("title", "")).casefold()
]
print("Constitution chunks:", len(chunks))


def find(phrase: str, width: int = 140, max_hits: int = 6) -> None:
    print(f"\n=== Chunks containing {phrase!r} ===")
    hits = 0
    for meta, text in chunks:
        position = text.casefold().find(phrase.casefold())
        if position < 0:
            continue
        hits += 1
        if hits > max_hits:
            continue
        snippet = " ".join(text[max(0, position - width): position + width].split())
        print(
            f"  label={meta.get('section_label')!r} chunk={meta.get('chunk_index')} "
            f"page={meta.get('page_start')}\n      ...{snippet}..."
        )
    print(f"  ({hits} chunk(s) in total)")


find("Equality before law")                      # Article 14
find("Protection of life and personal liberty")  # Article 21
find("Right to education")                       # Article 21A
find("Power of High Courts to issue certain writs")  # Article 226
