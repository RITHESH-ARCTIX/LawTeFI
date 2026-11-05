"""Remove the hardcoded "Writ Petition (Criminal) No. 167 of 2012" fact from generation.py.

Run from the project root (D:\\LawTeFI):
    python apply_docket_fix.py
or point it at another file:
    python apply_docket_fix.py path\\to\\generation.py

Safety: it makes a backup in your temp folder, and it changes nothing unless
every edit matches exactly once and the patched file still compiles.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "api/app/generation.py")
raw = path.read_bytes().decode("utf-8")
uses_crlf = "\r\n" in raw
text = raw.replace("\r\n", "\n")

FLAGS = re.M | re.S
EDITS = [
    (
        "remove caption_fact_extracted declaration",
        r"^ *caption_fact_extracted = False\n",
        "",
    ),
    (
        "replace hardcoded facts branch",
        r'( *)if field == "facts":\n *fact_match = re\.search\(.*?\n *else:\n *excerpt = source_text\[:260\]\n',
        r'\1if field == "facts":\n\1    excerpt = source_text[:260]\n',
    ),
    (
        "simplify truncation condition",
        r'not \(field == "facts" and caption_fact_extracted\) and ',
        "",
    ),
    (
        "replace caption if/else with a single evidence line",
        r"( *)if field == \"facts\" and fact_match:\n[^\n]*\n *else:\n *(fallback_lines\.append[^\n]*)\n",
        r"\1\2\n",
    ),
    (
        "simplify missing-section check",
        r"( *)if field != \"facts\" or not caption_fact_extracted:\n *if field not in missing:\n *missing\.append\(field\)\n",
        r"\1if field not in missing:\n\1    missing.append(field)\n",
    ),
    (
        "delete standalone docket block",
        r"^ *# The docket is the reliable procedural background.*?(?=^ *# Prefer the court's own enumerated concluding order)",
        "",
    ),
]

for name, pattern, replacement in EDITS:
    text, count = re.subn(pattern, replacement, text, flags=FLAGS)
    if count != 1:
        sys.exit(
            f"STOPPED: '{name}' matched {count} times (expected 1). "
            f"Nothing was changed. Send me this message and the lines around it."
        )
    print(f"ok  {name}")

leftovers = [word for word in ("167 of 2012", "fact_match", "caption_fact_extracted") if word in text]
if leftovers:
    sys.exit(f"STOPPED: still found {leftovers}. Nothing was changed.")

try:
    compile(text, str(path), "exec")
except SyntaxError as exc:
    sys.exit(f"STOPPED: patched file would not compile ({exc}). Nothing was changed.")

backup = Path(tempfile.gettempdir()) / (path.name + ".bak")
backup.write_bytes(raw.encode("utf-8"))
output = text.replace("\n", "\r\n") if uses_crlf else text
path.write_bytes(output.encode("utf-8"))
print(f"\nDone. {path} patched. Backup of the original: {backup}")
