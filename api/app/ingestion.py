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


def is_legal_heading(first_line: str) -> bool:
    if re.match(r"^(?:section|sec\.?|article)\s+\d+|^(?:PART|CHAPTER)\s+[IVXLCDM0-9]+", first_line, re.I):
        return True
    match = NUMBERED_SECTION_RE.match(first_line)
    if not match:
        return False
    title = match.group(2).strip()
    # Avoid treating amendment footnotes such as "1. 1st day of July..." as section 1.
    if re.match(r"\d+(?:st|nd|rd|th)\b", title, re.I):
        return False
    return len(title) <= 400


def legal_heading_labels(first_line: str) -> tuple[str, str]:
    """Return a concise citation anchor and a readable heading for statute text."""
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


@dataclass
class TextBlock:
    text: str
    page: int | None
    section_label: str | None = None
    paragraph_number: str | None = None
    heading: str | None = None


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
    # Join words split by line-end hyphenation, but preserve the remaining line breaks.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

    blocks: list[TextBlock] = []
    current_section: str | None = None
    current_heading: str | None = None
    for text, page in raw_blocks:
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
            if document_type == "constitution" and section_label.startswith("Section "):
                section_label = section_label.replace("Section ", "Article ", 1)
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
        anchor = next((item for item in reversed(current) if item.section_label or item.paragraph_number), current[-1])
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
        begins_unit = bool(block.paragraph_number or (block.section_label and block.text.startswith(block.section_label)))
        if current and (begins_unit or current_chars + len(block.text) + 2 > MAX_CHARS):
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

    # Upsert makes repeating the same manifest/file pair safe and keeps older file
    # versions as separate records because the source hash is part of the ID.
    for start in range(0, len(chunks), CHROMA_WRITE_BATCH_SIZE):
        end = start + CHROMA_WRITE_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=[chunk.text for chunk in chunks[start:end]],
            embeddings=vectors[start:end],
            metadatas=metadatas[start:end],
        )
    return len(chunks)


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
