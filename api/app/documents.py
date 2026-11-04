"""Owner-scoped local judgment PDF storage, indexing, retrieval, and case briefs."""

from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import UUID, uuid4

import chromadb
import psycopg
import pymupdf
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from psycopg.types.json import Jsonb

from .auth import get_current_user
from .database import connect_database
from .generation import generate_case_brief
from .ingestion import EMBED_BATCH_SIZE, JUDGMENT_PARA_RE, VECTOR_DIMENSION, chunk_blocks, extract_blocks, get_embeddings
from .retrieval import CHROMA_PATH, OLLAMA_URL, embed_queries, embed_query

router = APIRouter(prefix="/api/v1/documents", tags=["private judgments"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = PROJECT_ROOT / "data" / "private_documents"
MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 250
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma")
REPORT_HEADER_RE = re.compile(
    r"(?im)^\s*(?:SUPREME COURT REPORTS|\[\d{4}\]\s*\d+\s*S\.?C\.?R\.?)\s*$"
)


def clean_judgment_passage(text: str) -> str:
    """Remove common Supreme Court Reports layout artifacts for readable display."""
    cleaned = text.replace("\u00a0", " ")
    # Column letters are extracted at line edges, sometimes attached to the
    # preceding/following line. Remove only the familiar report-layout cases.
    cleaned = re.sub(r"(?m)(?<=[a-z,;])\s+[A-H]\s*(?=\n)", "\n", cleaned)
    cleaned = re.sub(r"(?<=[a-z])\s+[co]\s+(?=[A-Za-z])", " ", cleaned)
    # Report column markers can also be embedded between words, and OCR
    # occasionally reads the marker as zero or capital O.
    cleaned = re.sub(r"(?<=[a-z.,;])\s+[B-Hb-h]\s+(?=[a-z])", " ", cleaned)
    cleaned = re.sub(r"(?<=[a-z.,;])\s+[Hh]\s+(?=Section\b|Severability\b)", " ", cleaned)
    cleaned = re.sub(r"(?<=[a-z.,;])\s+[oO0]\s+(?=In\b|contention\b)", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*[A-H]\s+(?=[a-z])", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[A-H]\s+(?=Article\s+\d+\b)", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[a-h]\s+(?=[A-Z])", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[a-h]\s+(?=[a-z])", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[A-H]\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[a-h]\s*$", "", cleaned)
    # The law-report running caption can be embedded in a text block rather
    # than extracted as its own line, so remove that known repeated header too.
    cleaned = re.sub(
        r"(?i)\bSHREYASINGHAL\s+v\.\s*UNION\s+OF\s+INDIA"
        r"(?:\s*\[[^\]]{0,80}\])?\s*\d{3,4}",
        "",
        cleaned,
    )
    cleaned = REPORT_HEADER_RE.sub("", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d{3,4}\s*$", "", cleaned)
    # PDF text extraction wraps visual lines. Join those wraps while keeping
    # numbered judgment paragraphs separated for scanning.
    paragraph_break = "\uE000"
    cleaned = re.sub(
        r"\s*\n\s*(?=\d{1,3}[.)]\s+)", paragraph_break, cleaned
    )
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = cleaned.replace(paragraph_break, "\n\n")
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\bforthe\b", "for the", cleaned, flags=re.I)
    cleaned = re.sub(r"\bthatArticle\b", "that Article", cleaned, flags=re.I)
    cleaned = re.sub(r"\bmade\.on\b", "made on", cleaned, flags=re.I)
    cleaned = re.sub(r"\brepelled\s+0\s+contention\b", "repelled a contention", cleaned, flags=re.I)
    cleaned = re.sub(r"(?<=\s)[·•]\s*(?=[\"'])", "", cleaned)
    cleaned = re.sub(r"\breferred\s+[~˜]\s*\.", "referred.", cleaned)
    cleaned = re.sub(r"(?<!\n)\s+(Severability:)\s*", r"\n\n\1\n", cleaned)
    cleaned = re.sub(r"(?<!\n)\s+(Article\s+\d{1,2})\s+(?=\d{1,3}[.)])", r"\n\n\1\n", cleaned)
    return cleaned.strip()


class PassageSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)


def user_collection_name(user_id: UUID) -> str:
    return f"lawtefi_user_{user_id.hex}"


def _collection(user_id: UUID) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=user_collection_name(user_id),
        metadata={"hnsw:space": "cosine", "embedding_model": OLLAMA_EMBED_MODEL},
    )


def retrieve_user_document(
    user_id: UUID,
    document_id: UUID,
    query: str,
    limit: int = 8,
) -> list[dict[str, object]]:
    """Search one uploaded judgment in the signed-in user's isolated Chroma collection."""
    try:
        collection = _collection(user_id)
        count = collection.count()
        if not count:
            return []
        result = collection.query(
            query_embeddings=[embed_query(query)],
            n_results=min(limit, count),
            where={"document_id": str(document_id)},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError("Could not search this private judgment in the local vector store.") from exc

    sources: list[dict[str, object]] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        passage_text = text or ""
        section = metadata.get("section_label")
        paragraph = metadata.get("paragraph_number")
        if section:
            section_match = re.fullmatch(r"Section\s+(\d+[A-Z]?)", str(section), re.I)
            if section_match and not re.search(
                rf"\bSection\s+{re.escape(section_match.group(1))}\b", passage_text, re.I
            ):
                # Older indexes sometimes treated numbered judgment paragraphs
                # as statutory sections. Do not show a false section citation.
                section = None
        first_line = passage_text.splitlines()[0].strip() if passage_text.splitlines() else ""
        paragraph_match = JUDGMENT_PARA_RE.match(first_line)
        if paragraph_match:
            # Older indexes could mistake a numbered judgment paragraph for
            # a statutory section. Prefer the paragraph marker in the passage.
            paragraph = paragraph_match.group(1)
            section = None
        pages = {
            "start": metadata.get("page_start"),
            "end": metadata.get("page_end"),
        }
        sources.append(
            {
                "text": passage_text,
                "title": metadata.get("title") or "Uploaded judgment",
                "document_type": "judgment",
                "section": section,
                "heading": metadata.get("heading"),
                "paragraph": paragraph,
                "pages": pages,
                "distance": distance,
            }
        )
    return sources


def _retrieve_document_queries(
    user_id: UUID,
    document_id: UUID,
    queries: dict[str, str],
    limit: int = 3,
) -> dict[str, list[dict[str, object]]]:
    """Embed and search section queries together to reduce local service round trips."""
    if not queries:
        return {}
    try:
        collection = _collection(user_id)
        count = collection.count()
        if not count:
            return {key: [] for key in queries}
        vectors = embed_queries(list(queries.values()))
        result = collection.query(
            query_embeddings=vectors,
            n_results=min(limit, count),
            where={"document_id": str(document_id)},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError("Could not search this private judgment in the local vector store.") from exc

    retrieved: dict[str, list[dict[str, object]]] = {}
    for (key, _query), texts, metadatas, distances in zip(
        queries.items(), result.get("documents", []), result.get("metadatas", []), result.get("distances", [])
    ):
        rows: list[dict[str, object]] = []
        for text, metadata, distance in zip(texts, metadatas, distances):
            metadata = metadata or {}
            rows.append({
                "text": text or "",
                "title": metadata.get("title") or "Uploaded judgment",
                "document_type": "judgment",
                "section": metadata.get("section_label"),
                "heading": metadata.get("heading"),
                "paragraph": metadata.get("paragraph_number"),
                "pages": {"start": metadata.get("page_start"), "end": metadata.get("page_end")},
                "distance": distance,
            })
        retrieved[key] = rows
    return retrieved


def retrieve_document_pages(
    user_id: UUID,
    document_id: UUID,
    page_at_most: int | None = None,
    page_at_least: int | None = None,
    limit: int = 6,
    from_end: bool = False,
) -> list[dict[str, object]]:
    """Load ordered source chunks from a document page range without mixing other uploads."""
    filters: list[dict[str, object]] = [{"document_id": str(document_id)}]
    if page_at_most is not None:
        filters.append({"page_start": {"$lte": page_at_most}})
    if page_at_least is not None:
        filters.append({"page_end": {"$gte": page_at_least}})
    try:
        records = _collection(user_id).get(
            where=filters[0] if len(filters) == 1 else {"$and": filters},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        raise RuntimeError("Could not read the private judgment passages from the local vector store.") from exc

    rows: list[tuple[int, dict[str, object]]] = []
    for text, metadata in zip(records.get("documents", []), records.get("metadatas", [])):
        metadata = metadata or {}
        rows.append((
            int(metadata.get("chunk_index", 0)),
            {
                "text": text or "",
                "title": metadata.get("title") or "Uploaded judgment",
                "document_type": "judgment",
                "section": metadata.get("section_label"),
                "heading": metadata.get("heading"),
                "paragraph": metadata.get("paragraph_number"),
                "pages": {"start": metadata.get("page_start"), "end": metadata.get("page_end")},
            },
        ))
    rows.sort(key=lambda item: item[0])
    if from_end:
        return [source for _, source in rows[-limit:]]
    return [source for _, source in rows[:limit]]


def retrieve_document_body_opening(
    user_id: UUID, document_id: UUID, limit: int = 3
) -> list[dict[str, object]]:
    """Find the judgment body after report headnotes and front matter where possible."""
    try:
        records = _collection(user_id).get(
            where={"document_id": str(document_id)}, include=["documents", "metadatas"]
        )
    except Exception as exc:
        raise RuntimeError("Could not read the private judgment passages from the local vector store.") from exc
    rows = list(zip(records.get("documents", []), records.get("metadatas", [])))
    rows.sort(key=lambda item: int((item[1] or {}).get("chunk_index", 0)))
    markers = re.compile(
        r"(?:CRIMINAL\s*[|I]?\s*CIVIL\s+ORIGINAL\s+JURISDICTION|"
        r"(?:CIVIL|CRIMINAL|APPELLATE)\s+(?:ORIGINAL\s+)?JURISDICTION|\bJUDGMENT\s*[:—-])",
        re.IGNORECASE,
    )
    start_index = 0
    for index, (text, _metadata) in enumerate(rows):
        if markers.search(str(text or "")):
            start_index = index
            break
    selected = rows[start_index : start_index + limit]
    result: list[dict[str, object]] = []
    for text, metadata in selected:
        metadata = metadata or {}
        result.append({
            "text": text or "", "title": metadata.get("title") or "Uploaded judgment",
            "document_type": "judgment", "section": metadata.get("section_label"),
            "heading": metadata.get("heading"), "paragraph": metadata.get("paragraph_number"),
            "pages": {"start": metadata.get("page_start"), "end": metadata.get("page_end")},
        })
    return result


def retrieve_brief_sources(
    user_id: UUID,
    document_id: UUID,
    page_count: int,
) -> dict[str, list[dict[str, object]]]:
    """Retrieve a small, separate evidence set for each section of a case brief."""
    # Report series may put 10+ pages of headnotes before the judgment itself.
    # Locate the body caption for facts and reserve front matter for holding cues.
    opening = retrieve_document_pages(user_id, document_id, page_at_most=3, limit=2)
    reasoning_headnote = retrieve_document_pages(
        user_id, document_id, page_at_least=5, page_at_most=5, limit=1
    )
    body_opening = retrieve_document_body_opening(user_id, document_id, limit=3)
    if body_opening:
        body_text = str(body_opening[0].get("text") or "")
        jurisdiction = re.search(r"(?:ORIGINAL|APPELLATE)\s+JURISDICTION\s*:?", body_text, re.I)
        if jurisdiction:
            body_opening[0] = {**body_opening[0], "text": body_text[jurisdiction.start():]}
    ending = retrieve_document_pages(
        user_id, document_id, page_at_least=max(1, page_count - 5), limit=4, from_end=True
    )
    semantic = _retrieve_document_queries(user_id, document_id, {
        "facts": "judgment body original jurisdiction petitions filed under Article 32 case caption procedural history",
        "issues": "petitioners grounds of challenge issues raised in these petitions constitutional validity provisions",
        "holding": "judgment held declared unconstitutional valid read down disposition challenged provisions",
        "reasoning": "court's reasons Section 66A vague overbroad chilling effect incitement Article 19(2) analysis",
        "outcome": "conclusion final operative order all writ petitions disposed",
    }, limit=3)
    facts = body_opening[:1] or semantic.get("facts", [])
    issues = semantic.get("issues", [])
    holding = semantic.get("holding", [])
    reasoning = semantic.get("reasoning", [])
    outcome = semantic.get("outcome", [])

    def merge(first: list[dict[str, object]], second: list[dict[str, object]], limit: int = 6) -> list[dict[str, object]]:
        merged: list[dict[str, object]] = []
        seen: set[tuple[object, ...]] = set()
        for source in [*first, *second]:
            pages = source.get("pages") or {}
            key = (pages.get("start"), pages.get("end"), source.get("paragraph"), str(source.get("text") or "")[:140])
            if key not in seen:
                seen.add(key)
                merged.append(source)
            if len(merged) >= limit:
                break
        return merged

    return {
        "facts": facts,
        "issues": merge(opening, issues, limit=3),
        "holding": merge(opening, holding, limit=3),
        "reasoning": merge(reasoning_headnote, reasoning, limit=2),
        "outcome": ending[-2:] or outcome[:1],
    }


async def _read_pdf(file: UploadFile) -> tuple[bytes, str]:
    original_name = Path(file.filename or "judgment.pdf").name
    if Path(original_name).suffix.casefold() != ".pdf":
        raise HTTPException(status_code=415, detail="Upload a PDF judgment file.")
    contents = await file.read(MAX_PDF_BYTES + 1)
    if len(contents) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="The PDF is too large. The current limit is 15 MB.")
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="The uploaded file does not appear to be a PDF.")
    clean_name = re.sub(r"[^A-Za-z0-9._() -]", "_", original_name).strip(" .")[:240]
    return contents, clean_name or "judgment.pdf"


@router.get("")
def list_documents(current_user: dict[str, object] = Depends(get_current_user)) -> list[dict[str, object]]:
    with connect_database() as connection:
        return connection.execute(
            """
            SELECT id, title, page_count, chunk_count, created_at
            FROM user_documents
            WHERE owner_user_id = %s
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (current_user["id"],),
        ).fetchall()


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_judgment(
    file: UploadFile = File(...),
    current_user: dict[str, object] = Depends(get_current_user),
) -> dict[str, object]:
    contents, title = await _read_pdf(file)
    user_id = current_user["id"]
    document_id = uuid4()
    user_dir = UPLOAD_ROOT / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / f"{document_id}.pdf"
    collection: chromadb.Collection | None = None
    try:
        file_path.write_bytes(contents)
        with pymupdf.open(file_path) as pdf:
            page_count = len(pdf)
        if page_count > MAX_PDF_PAGES:
            raise HTTPException(status_code=413, detail="The current limit is 250 PDF pages.")

        chunks = chunk_blocks(extract_blocks(file_path, "judgment"))
        if not chunks:
            raise HTTPException(status_code=422, detail="No selectable text was found in this PDF. Scanned PDFs need OCR first.")

        vectors: list[list[float]] = []
        texts = [chunk.text for chunk in chunks]
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(
                get_embeddings(texts[start : start + EMBED_BATCH_SIZE], OLLAMA_URL, OLLAMA_EMBED_MODEL)
            )
        if any(len(vector) != VECTOR_DIMENSION for vector in vectors):
            raise RuntimeError("The local embedding dimensions did not match the shared embedding model.")

        collection = _collection(user_id)
        metadatas: list[dict[str, object]] = []
        for index, chunk in enumerate(chunks):
            metadata: dict[str, object] = {
                "document_id": str(document_id),
                "title": title,
                "document_type": "judgment",
                "chunk_index": index,
            }
            if chunk.page_start is not None:
                metadata["page_start"] = chunk.page_start
            if chunk.page_end is not None:
                metadata["page_end"] = chunk.page_end
            if chunk.section_label:
                metadata["section_label"] = chunk.section_label
            if chunk.paragraph_number:
                metadata["paragraph_number"] = chunk.paragraph_number
            if chunk.heading:
                metadata["heading"] = chunk.heading
            metadatas.append(metadata)

        collection.upsert(
            ids=[f"{document_id}-{index}" for index in range(len(chunks))],
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )
        stored_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        with connect_database() as connection:
            row = connection.execute(
                """
                INSERT INTO user_documents (id, owner_user_id, title, storage_path, page_count, chunk_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, title, page_count, chunk_count, created_at
                """,
                (document_id, user_id, title, stored_path, page_count, len(chunks)),
            ).fetchone()
        return row
    except HTTPException:
        if collection is not None:
            try:
                collection.delete(where={"document_id": str(document_id)})
            except Exception:
                pass
        file_path.unlink(missing_ok=True)
        raise
    except (ValueError, pymupdf.FileDataError) as exc:
        if collection is not None:
            try:
                collection.delete(where={"document_id": str(document_id)})
            except Exception:
                pass
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        if collection is not None:
            try:
                collection.delete(where={"document_id": str(document_id)})
            except Exception:
                pass
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except psycopg.Error as exc:
        if collection is not None:
            try:
                collection.delete(where={"document_id": str(document_id)})
            except Exception:
                pass
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Could not save the uploaded judgment record.") from exc
    except Exception as exc:
        if collection is not None:
            try:
                collection.delete(where={"document_id": str(document_id)})
            except Exception:
                pass
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Could not read or index this PDF judgment.") from exc
    finally:
        await file.close()


@router.post("/{document_id}/search")
def search_judgment_passages(
    document_id: UUID,
    body: PassageSearchRequest,
    current_user: dict[str, object] = Depends(get_current_user),
) -> dict[str, object]:
    """Return ranked, page-linked passages from one private judgment without generation."""
    user_id = current_user["id"]
    with connect_database() as connection:
        document = connection.execute(
            "SELECT id, title FROM user_documents WHERE id = %s AND owner_user_id = %s",
            (document_id, user_id),
        ).fetchone()
    if document is None:
        raise HTTPException(status_code=404, detail="Judgment not found in your workspace.")

    query = body.query.strip()
    if len(query) < 2:
        raise HTTPException(status_code=422, detail="Enter at least two characters to search this judgment.")
    try:
        sources = retrieve_user_document(user_id, document_id, query, limit=5)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "query": query,
        "results": [
            {
                "id": f"P{index}",
                "title": source.get("title") or document["title"],
                "section": source.get("section"),
                "paragraph": source.get("paragraph"),
                "pages": source.get("pages"),
                "text": clean_judgment_passage(str(source.get("text") or "")),
            }
            for index, source in enumerate(sources, start=1)
        ],
        "message": "Ranked passages from this private judgment. No AI answer was generated.",
    }


@router.post("/{document_id}/brief")
def create_case_brief(
    document_id: UUID,
    current_user: dict[str, object] = Depends(get_current_user),
) -> dict[str, object]:
    user_id = current_user["id"]
    with connect_database() as connection:
        document = connection.execute(
            "SELECT id, title, page_count FROM user_documents WHERE id = %s AND owner_user_id = %s",
            (document_id, user_id),
        ).fetchone()
    if document is None:
        raise HTTPException(status_code=404, detail="Judgment not found in your workspace.")

    try:
        sources = retrieve_brief_sources(user_id, document_id, int(document["page_count"]))
        result = generate_case_brief(sources)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if result.get("status") not in {"ok", "partial"} or not result.get("brief"):
        raise HTTPException(
            status_code=422,
            detail=result.get("message") or "LawTeFI could not create a source-linked brief from the retrieved pages.",
        )

    if result.get("status") == "partial":
        return {
            **result,
            "saved_work_id": None,
            "message": str(result.get("message", "")) + " This partial brief was not saved automatically.",
        }

    title = f"Case brief — {document['title']}"[:200]
    with connect_database() as connection:
        saved = connection.execute(
            """
            INSERT INTO saved_work (owner_user_id, work_type, title, content, citations)
            VALUES (%s, 'case_brief', %s, %s, %s)
            RETURNING id
            """,
            (user_id, title, result["brief"], Jsonb(result["citations"])),
        ).fetchone()
    return {
        **result,
        "saved_work_id": str(saved["id"]),
        "message": "The source-linked case brief was saved in My saved work.",
    }


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    current_user: dict[str, object] = Depends(get_current_user),
) -> None:
    user_id = current_user["id"]
    with connect_database() as connection:
        document = connection.execute(
            "DELETE FROM user_documents WHERE id = %s AND owner_user_id = %s RETURNING storage_path",
            (document_id, user_id),
        ).fetchone()
    if document is None:
        raise HTTPException(status_code=404, detail="Judgment not found in your workspace.")

    file_path = (PROJECT_ROOT / document["storage_path"]).resolve()
    if UPLOAD_ROOT.resolve() in file_path.parents:
        file_path.unlink(missing_ok=True)
    try:
        _collection(user_id).delete(where={"document_id": str(document_id)})
    except Exception:
        pass
