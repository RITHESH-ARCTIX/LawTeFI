"""Show how section/article labels are stored in LawTeFI's Chroma index.

Put this file in D:\\LawTeFI\\api, activate the venv, then run:
    python check_labels.py
It only reads the index; nothing is changed. Paste the output back to Claude.
"""

from __future__ import annotations

import re
from collections import Counter

from app.retrieval import _collection

metas = [m or {} for m in _collection().get(include=["metadatas"])["metadatas"]]
print("Total chunks:", len(metas))
for title, count in Counter(m.get("title") for m in metas).items():
    print(f"  {count:5}  {title}")


def show_labels(title_part: str, numbers: list[str]) -> None:
    print(f"\nChunks in '{title_part}' whose label starts with one of {numbers}:")
    pattern = re.compile(
        r"^(?:article\s+|section\s+)?(" + "|".join(numbers) + r")(?![0-9A-Za-z])", re.I
    )
    found = False
    for m in metas:
        if title_part.casefold() not in str(m.get("title", "")).casefold():
            continue
        label = str(m.get("section_label") or "")
        if pattern.match(label):
            found = True
            print(
                f"   label={label[:60]!r} | type={m.get('document_type')} "
                f"| chunk={m.get('chunk_index')} | page={m.get('page_start')}"
            )
    if not found:
        print("   (none found)")


show_labels("Constitution", ["14", "17", "19", "21", "21A", "226"])
show_labels("Bharatiya", ["103", "104"])

constitution = [m for m in metas if "constitution" in str(m.get("title", "")).casefold()]
print("\nFirst 15 Constitution labels:", [str(m.get("section_label"))[:40] for m in constitution[:15]])
print(
    "Constitution chunks with no label:",
    sum(1 for m in constitution if not m.get("section_label")),
    "of",
    len(constitution),
)
