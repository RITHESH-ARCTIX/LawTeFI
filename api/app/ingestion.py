"""Local legal-document ingestion: extract, structure, chunk, embed, and persist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import httpx
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "source_docs"
DEFAULT_MANIFEST = ROOT / "data" / "manifest.jsonl"
DEFAULT_CHROMA_PATH = ROOT / "data" / "chroma"
MAX_CHARS = 1800
OVERLAP_CHARS = 180
EMBED_BATCH_SIZE = 24
CHROMA_WRITE_BATCH_SIZE = 100
VECTOR_DIMENSION = 768  # embeddinggemma; must match the collection and query vectors

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
SECTION_RE = re.compile(
    r"^(?:(?:section|sec\.?|article)\s+\d+[A-Z]?(?:\([^)]+\))?|"
    r"(?:PART|CHAPTER)\s+[IVXLCDM0-9]+|"
    r"\d+[A-Z]?(?:\.\d+)*\s*[.)-]\s+\S)",
    re.IGNORECASE,
)
# OCR can prefix judgment paragraph numbers with stray column markers (for
# example, "· 83."). Ignore a few leading non-digits when recognizing them.
JUDGMENT_PARA_RE = re.compile(r"^\s*[^\d]{0,12}(\d{1,3})\s*\)?[.)]\s+\S")
NUMBERED_SECTION_RE = re.compile(r"^(\d+[A-Z]?(?:\.\d+)*\s*[.)])\s+(.+)$", re.I)

# Official Acts and the Constitution print every heading as
# "14. Equality before law.—The State shall not deny ...". In the PDF text layer a
# heading is often in the MIDDLE of a text block, may carry an amendment marker
# ("2[21A. Right to education.—..."), and may wrap over two lines. So headings are
# searched for anywhere in the text instead of only at the start of a block.
# Match section numbers followed either by:
# 1) A title and dash: "14. Equality before law.—" or "2[21A. Right to education.—"
# 2) A subsection parenthetical directly: "14. (1)" or "14. (a)"
# 3) A title followed by a period/colon and subsection: "14. Short title. (1)"
# Match section numbers followed by either:
# Matches:
# Universal section heading matcher for Indian statutes & Constitution:
# Matches:
# 1) "14. Equality before law.—" or "104. Penalty for sitting and voting..."
# 2) "14. (1)" or "1. (a)"
# 3) Headings with amendment markers: "2[21A. Right to education.—"
HEADING_INLINE_RE = re.compile(
    r"(?:(?<=\s)|^)(?:\d{1,2}\[)?(?P<number>\d{1,3}[A-Z]{0,2})\.\s+"
    r"(?:"
    r"(?=\(\d+[a-z]?\)|\([a-z]\))|"                                              # Direct (1) or (a)
    r"(?P<title>[A-Z][^\n\r]{1,160}?)(?:\.[\s—–]|\s?[—–]|\.-|\.\s+|$)"          # Title
    r")"
)
FOOTNOTE_IGNORE_RE = re.compile(
    r"^(?:Ins\b|Subs\b|Omitted\b|See\b|W\.e\.f\b|1st\b|2nd\b|3rd\b|\d+th\b|Act\s*[,(0-9]|Vide\b)",
    re.I,
)

# Schedules and appendices contain numbered paragraphs that look like headings.
BACK_MATTER_RE = re.compile(
    r"^(?:THE\s+)?(?P<name>(?:(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"ELEVENTH|TWELFTH)\s+)?SCHEDULE|APPENDIX\s+[IVXLC0-9]+)\b"
)
EXPLICIT_UNIT_RE = re.compile(
    r"^(?:(?:Section|SECTION|Sec\.?|SEC\.?|Article|ARTICLE)\s+\d+|(?:PART|CHAPTER|Part|Chapter)\s+[IVXLCDM0-9]+)"
)
CONTEXT_HEADING_RE = re.compile(r"^(?:PART|CHAPTER|Part|Chapter)\s+[IVXLCDM0-9]+")


def is_legal_heading(first_line: str) -> bool:
    if re.match(r"^(?:section|sec\.?|article)\s+\d+|^(?:PART|CHAPTER)\s+[IVXLCDM0-9]+", first_line, re.I):
        return True
    match = NUMBERED_SECTION_RE.match(first_line)
    if not match:
        return False
    title = match.group(2).strip()
    if re.match(r"\d+(?:st|nd|rd|th)\b", title, re.I):
        return False
    return len(title) <= 400


def is_explicit_unit(first_line: str, allow_plain_numbered: bool) -> bool:
    if EXPLICIT_UNIT_RE.match(first_line):
        return True
    return allow_plain_numbered and bool(SECTION_RE.match(first_line)) and is_legal_heading(first_line)


def legal_heading_labels(first_line: str) -> tuple[str, str]:
    numbered = NUMBERED_SECTION_RE.match(first_line)
    if numbered:
        section_number = numbered.group(1).rstrip(".)- ")
        return f"Section {section_number}", first_line[:220]

    explicit = re.match(r"^(section|sec\.?|article)\s+(\d+[A-Z]?(?:\([^)]+\))?)\s*[:.—-]?\s*(.*)$", first_line, re.I)
    if explicit:
        kind = "Article" if explicit.group(1).casefold().startswith("article") else "Section"
        label = f"{kind} {explicit.group(2)}"
        heading = explicit.group(3).strip() or first_line
        return label, heading[:220]

    label = first_line[:100]
    return label, first_line[:220]


def back_matter_label(match: re.Match[str]) -> str:
    name = " ".join(match.group("name").split())
    if name.startswith("APPENDIX"):
        return "Appendix " + name.split()[-1]
    return name.title()


def split_at_headings(text: str) -> list[tuple[str, str | None, str | None]]:
    """Split block text at numbered headings found anywhere inside it."""
    matches = list(HEADING_INLINE_RE.finditer(text))
    if not matches:
        return [(text, None, None)]

    valid_matches = []
    for m in matches:
        num = m.group("number")
        title = (m.group("title") or "").strip()
        # Ignore actual amendment footnotes
        if FOOTNOTE_IGNORE_RE.match(title):
            continue
        try:
            num_int = int(re.sub(r"[A-Z]", "", num))
            if num_int > 550 or num_int == 0:
                continue
        except ValueError:
            pass
        valid_matches.append(m)

    if not valid_matches:
        return [(text, None, None)]

    segments: list[tuple[str, str | None, str | None]] = []
    lead = text[: valid_matches[0].start()].strip()
    if lead:
        segments.append((lead, None, None))
    for index, match in enumerate(valid_matches):
        end = valid_matches[index + 1].start() if index + 1 < len(valid_matches) else len(text)
        segment = text[match.start():end].strip()
        number = match.group("number")
        title = (match.group("title") or "").strip()
        segments.append((segment, number, title or None))
    return segments

@dataclass
class TextBlock:
    text: str
    page: int | None
    section_label: str | None = None
    paragraph_number: str | None = None
    heading: str | None = None
    begins_unit: bool = False  # True where a new article/section/schedule starts


@dataclass
class LegalChunk:
    text: str
    page_start: int | None
    page_end: int | None
    section_label: str | None
    paragraph_number: str | None
    heading: str | None


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u00a0", " ").replace("\ufeff", "")
    # Join words split by line-end hyphenation, but preserve remaining line breaks.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def label_blocks(
    raw_blocks: list[tuple[str, int | None]], document_type: str, allow_plain_numbered: bool
) -> list[TextBlock]:
    """Attach article/section/paragraph labels to extracted text blocks."""
    blocks: list[TextBlock] = []
    current_section: str | None = None
    current_heading: str | None = None
    legal_text = document_type in {"statute", "constitution"}
    in_back_matter = False
    pending_context: list[str] = []

    for text, page in raw_blocks:
        if legal_text:
            first_line = text.splitlines()[0].strip()
            back_matter = BACK_MATTER_RE.match(first_line)
            if back_matter:
                in_back_matter = True
                current_section = back_matter_label(back_matter)
                current_heading = first_line[:220]
                blocks.append(
                    TextBlock(
                        text=text,
                        page=page,
                        section_label=current_section,
                        heading=current_heading,
                        begins_unit=True,
                    )
                )
                continue
            if in_back_matter:
                blocks.append(
                    TextBlock(
                        text=text,
                        page=page,
                        section_label=current_section,
                        heading=current_heading,
                    )
                )
                continue

            segments_with_headings = split_at_headings(text)
            has_split_heading = any(num is not None for _, num, _ in segments_with_headings)

            if has_split_heading:
                for segment, number, title in segments_with_headings:
                    if number:
                        kind = "Article" if document_type == "constitution" else "Section"
                        current_section = f"{kind} {number}"
                        if title:
                            clean_title = " ".join(str(title).split()).rstrip(" .—–-")
                            current_heading = f"{number}. {clean_title}."[:220]
                        else:
                            current_heading = f"{kind} {number}"

                        if pending_context:
                            segment = "\n".join(pending_context) + "\n\n" + segment
                            pending_context = []
                        blocks.append(
                            TextBlock(
                                text=segment,
                                page=page,
                                section_label=current_section,
                                heading=current_heading,
                                begins_unit=True,
                            )
                        )
                    else:
                        blocks.append(
                            TextBlock(
                                text=segment,
                                page=page,
                                section_label=current_section,
                                heading=current_heading,
                            )
                        )
                continue

            # Fallback: check if the block begins with a numbered section: e.g. "14. ..." or "104. ..."
            num_match = re.match(r"^(?:(?:\d{1,2}\[)?(?P<num>\d{1,3}[A-Z]{0,2}))\.\s+(?P<rest>.+)$", first_line)
            if num_match and not FOOTNOTE_IGNORE_RE.match(num_match.group("rest").strip()):
                num_str = num_match.group("num")
                try:
                    num_val = int(re.sub(r"[A-Z]", "", num_str))
                    if 0 < num_val <= 550:
                        kind = "Article" if document_type == "constitution" else "Section"
                        current_section = f"{kind} {num_str}"
                        clean_title = num_match.group("rest").split("—")[0].split(".")[0].strip()
                        current_heading = f"{num_str}. {clean_title}."[:220]
                        if pending_context:
                            text = "\n".join(pending_context) + "\n\n" + text
                            pending_context = []
                        blocks.append(
                            TextBlock(
                                text=text,
                                page=page,
                                section_label=current_section,
                                heading=current_heading,
                                begins_unit=True,
                            )
                        )
                        continue
                except ValueError:
                    pass

            if CONTEXT_HEADING_RE.match(first_line):
                pending_context.append(" ".join(text.split()))
                continue

            if is_explicit_unit(first_line, allow_plain_numbered):
                section_label, heading = legal_heading_labels(first_line)
                if document_type == "constitution" and section_label.startswith("Section "):
                    section_label = section_label.replace("Section ", "Article ", 1)
                current_section, current_heading = section_label, heading
                if pending_context:
                    text = "\n".join(pending_context) + "\n\n" + text
                    pending_context = []
                blocks.append(
                    TextBlock(
                        text=text,
                        page=page,
                        section_label=current_section,
                        heading=current_heading,
                        begins_unit=True,
                    )
                )
                continue

            blocks.append(
                TextBlock(
                    text=text,
                    page=page,
                    section_label=current_section,
                    heading=current_heading,
                )
            )
            continue

        # Judgments and other document types
        first_line = text.splitlines()[0].strip()
        section_label = None
        paragraph_number = None
        heading = None
        if document_type == "judgment":
            para_match = JUDGMENT_PARA_RE.match(first_line)
            if para_match:
                paragraph_number = para_match.group(1)
                section_label = current_section
                heading = current_heading
            elif SECTION_RE.match(first_line) and is_legal_heading(first_line):
                section_label, heading = legal_heading_labels(first_line)
                current_heading = heading
                current_section = section_label
            else:
                section_label = current_section
                heading = current_heading
        elif SECTION_RE.match(first_line) and is_legal_heading(first_line):
            section_label, heading = legal_heading_labels(first_line)
            current_heading = heading
            current_section = section_label
        else:
            section_label = current_section
            heading = current_heading

        blocks.append(
            TextBlock(
                text=text,
                page=page,
                section_label=section_label,
                paragraph_number=paragraph_number,
                heading=heading,
            )
        )
    return blocks


def extract_blocks(path: Path, document_type: str) -> list[TextBlock]:
    """Extract text in reading order and retain printed PDF page locations."""
    raw_blocks: list[tuple[str, int | None]] = []
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                for block in page.get_text("blocks", sort=True):
                    text = normalize_text(str(block[4]))
                    # Gazette PDFs can expose image placeholders and legacy-font
                    # Hindi masthead text as garbled roman characters. This block
                    # is page furniture, not operative legal text.
                    if "xxxGIDHxxx" in text or "xxxGIDExxx" in text:
                        continue
                    if document_type == "constitution":
                        # India Code's English Constitution PDF uses a broken
                        # glyph for several em dashes in its text layer.
                        text = text.replace("�", "—")
                    if text and text.casefold() != "indiacode" and not re.fullmatch(r"\d{1,4}", text):
                        raw_blocks.append((text, page_number))
        if not raw_blocks:
            raise ValueError(
                f"No selectable text found in {path.name}. It may be a scanned PDF; OCR is not enabled."
            )
        if document_type == "statute":
            # Indian central Acts often place the full arrangement of sections before the Act.
            # Start at the enactment text so bare section headings from the table of contents
            # or Gazette masthead are not indexed as if they were operative law.
            first_body_block = next(
                (
                    index
                    for index, (text, _) in enumerate(raw_blocks)
                    if re.search(r"\bBE\s+it\s+enacted\s+by\s+Parliament\b", text, re.I)
                ),
                None,
            )
            if first_body_block is not None:
                raw_blocks = raw_blocks[first_body_block:]
        elif document_type == "constitution":
            # Skip cover, preface, abbreviations, and contents; begin with the
            # operative constitutional text while keeping printed PDF pages.
            with pymupdf.open(path) as pdf:
                first_body_page = next(
                    (
                        page_number
                        for page_number, page in enumerate(pdf, start=1)
                        if "WE, THE PEOPLE OF INDIA" in page.get_text("text", sort=True)
                    ),
                    None,
                )
            if first_body_page is not None:
                raw_blocks = [(text, page) for text, page in raw_blocks if page is None or page >= first_body_page]
    else:
        text = normalize_text(path.read_text(encoding="utf-8-sig"))
        # Form-feed page breaks in text exports are preserved as page references.
        for page_number, page_text in enumerate(text.split("\f"), start=1):
            for block in re.split(r"\n\s*\n", page_text):
                block = normalize_text(block)
                if block:
                    raw_blocks.append((block, page_number if "\f" in text else None))

    return label_blocks(raw_blocks, document_type, allow_plain_numbered=path.suffix.lower() != ".pdf")


def split_oversized(text: str, limit: int = MAX_CHARS) -> list[str]:
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        boundary = max(remaining.rfind(". ", 0, limit), remaining.rfind("; ", 0, limit))
        if boundary < limit // 2:
            boundary = remaining.rfind(" ", 0, limit)
        if boundary <= 0:
            boundary = limit
        else:
            boundary += 1
        piece = remaining[:boundary].strip()
        if piece:
            pieces.append(piece)
        next_start = max(0, boundary - OVERLAP_CHARS)
        remaining = remaining[next_start:].strip()
        if next_start == 0:
            remaining = remaining[boundary:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_blocks(blocks: list[TextBlock]) -> list[LegalChunk]:
    """Pack legal blocks without crossing section/paragraph boundaries unnecessarily."""
    chunks: list[LegalChunk] = []
    current: list[TextBlock] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        text = "\n\n".join(item.text for item in current).strip()
        page_numbers = [item.page for item in current if item.page is not None]
        # A chunk that starts a new article/section/schedule is labelled by that
        # heading. Judgment chunks keep the original "last labelled block" anchor.
        anchor = next((item for item in current if item.begins_unit), None) or next(
            (item for item in reversed(current) if item.section_label or item.paragraph_number), current[-1]
        )
        for piece in split_oversized(text):
            chunks.append(
                LegalChunk(
                    text=piece,
                    page_start=min(page_numbers) if page_numbers else None,
                    page_end=max(page_numbers) if page_numbers else None,
                    section_label=anchor.section_label,
                    paragraph_number=anchor.paragraph_number,
                    heading=anchor.heading,
                )
            )
        current = []
        current_chars = 0

    for block in blocks:
        # Start a fresh chunk when a new explicit legal unit begins.
        starts_new_unit = bool(
            block.begins_unit
            or block.paragraph_number
            or (block.section_label and block.text.startswith(block.section_label))
        )
        if current and starts_new_unit:
            flush()
        elif current and (current_chars + len(block.text) + 2 > MAX_CHARS):
            flush()

        if len(block.text) > MAX_CHARS:
            flush()
            current = [block]
            current_chars = len(block.text)
            flush()
            continue
        current.append(block)
        current_chars += len(block.text) + 2
    flush()
    return chunks


def load_manifest(path: Path, input_dir: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Copy data/manifest.example.jsonl to data/manifest.jsonl and update its source records."
        )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on manifest line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Manifest line {line_number} must be a JSON object.")
        required = ("file", "title", "document_type", "source_url")
        missing = [key for key in required if not record.get(key)]
        if missing:
            raise ValueError(f"Manifest line {line_number} is missing: {', '.join(missing)}")
        record["path"] = (input_dir / record["file"]).resolve()
        if input_dir.resolve() not in record["path"].parents:
            raise ValueError(f"Manifest file must be inside {input_dir}: {record['file']}")
        if record["path"].suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type for {record['file']}; use PDF, TXT, or Markdown.")
        if not record["path"].is_file():
            raise FileNotFoundError(f"Source file not found: {record['path']}")
        if record["document_type"] not in {"constitution", "statute", "judgment", "report"}:
            raise ValueError(f"Unsupported document_type on line {line_number}.")
        records.append(record)
    return records


def get_embeddings(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/embed",
        json={"model": model, "input": texts},
        timeout=180.0,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Embedding request failed ({response.status_code}). Ensure Ollama is running and the model '{model}' is available."
        )
    payload = response.json()
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError("Embedding service returned a different number of vectors than requested.")
    for vector in embeddings:
        if len(vector) != VECTOR_DIMENSION:
            raise RuntimeError(
                f"Embedding dimension is {len(vector)}, expected {VECTOR_DIMENSION}; update the configured model and database schema together."
            )
    return embeddings


def ingest_record(
    collection: chromadb.Collection,
    record: dict[str, Any],
    chunks: list[LegalChunk],
    vectors: list[list[float]],
    content_sha256: str,
) -> int:
    document_key = hashlib.sha256(f"{record['source_url']}:{content_sha256}".encode("utf-8")).hexdigest()
    metadatas: list[dict[str, str | int | float | bool]] = []
    ids: list[str] = []
    for index, chunk in enumerate(chunks):
        metadata: dict[str, str | int | float | bool] = {
            "document_id": document_key,
            "chunk_index": index,
            "title": record["title"],
            "document_type": record["document_type"],
            "jurisdiction": record.get("jurisdiction", "India"),
            "source_url": record["source_url"],
            "content_sha256": content_sha256,
            "language_code": record.get("language_code", "en"),
        }
        for key in ("issuing_authority", "case_name", "version_label", "published_on", "effective_from", "effective_until"):
            if record.get(key):
                metadata[key] = str(record[key])
        for key in ("section_label", "paragraph_number", "heading", "page_start", "page_end"):
            value = getattr(chunk, key)
            if value is not None:
                metadata[key] = value
        metadatas.append(metadata)
        ids.append(f"{document_key}-{index}")

    # Re-ingesting the same file replaces its earlier chunks, so a change in the
    # chunking rules cannot leave stale chunks behind. Other file versions keep
    # their own records because the source hash is part of document_id.
    collection.delete(where={"document_id": document_key})
    for start in range(0, len(chunks), CHROMA_WRITE_BATCH_SIZE):
        end = start + CHROMA_WRITE_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=[chunk.text for chunk in chunks[start:end]],
            embeddings=vectors[start:end],
            metadatas=metadatas[start:end],
        )
    return len(chunks)


def print_chunk_report(chunks: list[LegalChunk], find: list[str] | None) -> None:
    labelled = [chunk for chunk in chunks if chunk.section_label]
    print(
        f"  labelled chunks: {len(labelled)}/{len(chunks)}; "
        f"distinct labels: {len({chunk.section_label for chunk in labelled})}"
    )
    for number in find or []:
        wanted = {f"Article {number}", f"Section {number}"}
        hits = [chunk for chunk in chunks if chunk.section_label in wanted]
        if not hits:
            print(f"  [find {number}] no chunk is labelled Article/Section {number}")
        for chunk in hits[:3]:
            snippet = " ".join(chunk.text.split())[:110]
            print(f"  [find {number}] {chunk.section_label} p.{chunk.page_start}: {snippet}")


def main() -> int:
    # Keep the CLI usable in Windows terminals whose active code page cannot encode
    # all Unicode characters present in legal-source PDFs.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Ingest curated legal source files into the LawTeFI corpus.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true", help="Extract and chunk only; do not embed or write to PostgreSQL.")
    parser.add_argument("--only", nargs="+", help="Process only these manifest filenames (useful for reviewing one source at a time).")
    parser.add_argument("--find", nargs="+", help="Print the chunks labelled Article/Section N for each given N (e.g. --find 14 21 226).")
    parser.add_argument("--embedding-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--chroma-path", type=Path, default=Path(os.environ.get("CHROMA_PATH", str(DEFAULT_CHROMA_PATH))))
    args = parser.parse_args()

    try:
        records = load_manifest(args.manifest, args.input_dir)
        if args.only:
            selected = set(args.only)
            unknown = selected - {record["file"] for record in records}
            if unknown:
                raise ValueError(f"Files not found in manifest: {', '.join(sorted(unknown))}")
            records = [record for record in records if record["file"] in selected]
        if not records:
            print("No records found in the manifest.")
            return 0

        prepared: list[tuple[dict[str, Any], list[LegalChunk], str]] = []
        for record in records:
            blocks = extract_blocks(record["path"], record["document_type"])
            chunks = chunk_blocks(blocks)
            digest = hashlib.sha256(record["path"].read_bytes()).hexdigest()
            prepared.append((record, chunks, digest))
            print(f"{record['title']}: {len(blocks)} extracted blocks -> {len(chunks)} chunks")
            for chunk in chunks[:2]:
                anchor = chunk.section_label or (f"para. {chunk.paragraph_number}" if chunk.paragraph_number else "unlabeled")
                page = f", p. {chunk.page_start}" if chunk.page_start else ""
                print(f"  - {anchor}{page}: {chunk.text[:180].replace(chr(10), ' ')}")
            print_chunk_report(chunks, args.find)

        if args.dry_run:
            print("Dry run complete. Nothing was embedded or written to the database.")
            return 0

        all_text = [chunk.text for _, chunks, _ in prepared for chunk in chunks]
        all_vectors: list[list[float]] = []
        print(f"Generating {len(all_text)} embeddings with local model '{args.embedding_model}'.")
        for start in range(0, len(all_text), EMBED_BATCH_SIZE):
            end = min(start + EMBED_BATCH_SIZE, len(all_text))
            print(f"  Embedding chunks {start + 1}-{end} of {len(all_text)}...")
            all_vectors.extend(get_embeddings(all_text[start:end], args.ollama_url, args.embedding_model))

        args.chroma_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving to local Chroma storage at {args.chroma_path}.")
        client = chromadb.PersistentClient(path=str(args.chroma_path))
        collection = client.get_or_create_collection(
            name="lawtefi_legal",
            metadata={"hnsw:space": "cosine", "embedding_model": args.embedding_model},
        )
        stored_model = (collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != args.embedding_model:
            raise RuntimeError(
                f"The Chroma collection uses '{stored_model}', but this run uses '{args.embedding_model}'. "
                "Keep the same embedding model or create a fresh collection before re-indexing."
            )
        cursor = 0
        for record, chunks, digest in prepared:
            vectors = all_vectors[cursor : cursor + len(chunks)]
            cursor += len(chunks)
            count = ingest_record(collection, record, chunks, vectors, digest)
            print(f"Stored {count} chunks for {record['title']} in {args.chroma_path}.")
        print("Ingestion complete.")
        return 0
    except Exception as exc:  # CLI prints a useful failure without hiding the underlying reason.
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())