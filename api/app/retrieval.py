"""Retrieve relevant passages from LawTeFI's local legal corpus."""

from __future__ import annotations
from functools import lru_cache


import os
import re
from pathlib import Path
from typing import Any

import chromadb
import httpx


CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", str(Path(__file__).resolve().parents[2] / "data" / "chroma")))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "lawtefi_legal")
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma")
VECTOR_DIMENSION = 768
STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "by", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "say", "the", "to", "under",
    "what", "which", "who", "with", "bharatiya", "nyaya", "sanhita", "bns",
}

@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(COLLECTION_NAME)


def embed_query(query: str) -> list[float]:
    """Create a query vector using the same local Ollama model used for ingestion."""
    return embed_queries([query])[0]


def embed_queries(queries: list[str]) -> list[list[float]]:
    """Embed several queries in one local Ollama request."""
    if not queries:
        return []
    try:
        response = httpx.post(
            f"{OLLAMA_URL.rstrip('/')}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": queries},
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Could not reach the local embedding model at {OLLAMA_URL}. Make sure Ollama is running."
        ) from exc

    vectors = response.json().get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(queries):
        raise RuntimeError("The local embedding service returned an invalid response.")
    for vector in vectors:
        if len(vector) != VECTOR_DIMENSION:
            raise RuntimeError(
                f"The query vector has {len(vector)} dimensions; the indexed corpus expects {VECTOR_DIMENSION}."
            )
    return vectors


def citation_section(metadata: dict[str, Any]) -> str | None:
    """Normalize both existing and newly ingested section labels for display."""
    raw_label = metadata.get("section_label")
    if not raw_label:
        return None
    label = str(raw_label)
    explicit = re.match(r"^(?:section|sec\.?|article)\s+(\d+[A-Z]?(?:\([^)]+\))?)", label, re.I)
    if explicit:
        prefix = "Article" if label.casefold().startswith("article") else "Section"
        return f"{prefix} {explicit.group(1)}"
    numbered = re.match(r"^(\d+[A-Z]?(?:\.\d+)*)(?:\s*[.)-])?\s+", label)
    if numbered and metadata.get("document_type") == "statute":
        return f"Section {numbered.group(1)}"
    return label


def get_statute_section(title: str, section_number: str) -> list[dict[str, Any]]:
    """Load all indexed chunks for one statute section in document order."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        collection = client.get_collection(COLLECTION_NAME)
        records = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        raise RuntimeError(
            f"The legal index was not found at {CHROMA_PATH}. Run the ingestion step first."
        ) from exc

    section_pattern = re.compile(
        rf"^(?:section\s+)?{re.escape(section_number)}(?:\b|[.)—–-])",
        re.IGNORECASE,
    )
    selected: list[tuple[int, dict[str, Any]]] = []
    for text, metadata in zip(records.get("documents", []), records.get("metadatas", [])):
        metadata = metadata or {}
        if metadata.get("title") != title or metadata.get("document_type") != "statute":
            continue
        raw_section = str(metadata.get("section_label") or "")
        if not section_pattern.match(raw_section):
            continue
        selected.append(
            (
                int(metadata.get("chunk_index", 0)),
                {
                    "text": text or "",
                    "title": metadata.get("title"),
                    "document_type": metadata.get("document_type"),
                    "jurisdiction": metadata.get("jurisdiction"),
                    "section": citation_section(metadata),
                    "heading": metadata.get("heading"),
                    "paragraph": metadata.get("paragraph_number"),
                    "pages": {
                        "start": metadata.get("page_start"),
                        "end": metadata.get("page_end"),
                    },
                    "source_url": metadata.get("source_url"),
                    "distance": None,
                },
            )
        )
    selected.sort(key=lambda item: item[0])
    return [source for _, source in selected]


def list_corpus_sources() -> list[dict[str, Any]]:
    """Group indexed chunks into versioned source-document records for the catalog."""
    try:
        metadatas = _collection().get(include=["metadatas"]).get("metadatas", [])
    except Exception as exc:
        raise RuntimeError("The shared legal corpus is unavailable.") from exc

    sources: dict[str, dict[str, Any]] = {}
    for metadata in metadatas:
        metadata = metadata or {}
        document_id = str(metadata.get("document_id") or "")
        if not document_id:
            continue
        source = sources.setdefault(
            document_id,
            {
                "title": metadata.get("title") or "Untitled source",
                "document_type": metadata.get("document_type") or "source",
                "jurisdiction": metadata.get("jurisdiction") or "India",
                "issuing_authority": metadata.get("issuing_authority"),
                "language_code": metadata.get("language_code"),
                "version_label": metadata.get("version_label"),
                "published_on": metadata.get("published_on"),
                "effective_from": metadata.get("effective_from"),
                "effective_until": metadata.get("effective_until"),
                "source_url": metadata.get("source_url"),
                "chunk_count": 0,
                "page_start": None,
                "page_end": None,
            },
        )
        source["chunk_count"] += 1
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        if isinstance(page_start, int):
            current = source["page_start"]
            source["page_start"] = page_start if current is None else min(current, page_start)
        if isinstance(page_end, int):
            current = source["page_end"]
            source["page_end"] = page_end if current is None else max(current, page_end)

    return sorted(
        sources.values(),
        key=lambda source: (str(source["title"]).casefold(), str(source.get("version_label") or "")),
    )


def retrieve(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return the closest source chunks, including their legal citation metadata."""
    try:
        collection = _collection()
    except Exception as exc:
        raise RuntimeError(
            f"The legal index was not found at {CHROMA_PATH}. Run the ingestion step first."
        ) from exc

    if collection.count() == 0:
        return []

    vector = embed_query(query)
    candidate_count = min(max(limit * 4, 20), collection.count())
    result = collection.query(
        query_embeddings=[vector],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    query_terms = {
        term for term in re.findall(r"[a-z0-9]+", query.casefold())
        if len(term) > 1 and term not in STOP_WORDS
    }
    definition_intent = bool(
        re.search(r"\b(?:what\s+(?:is|does)|define|definition|meaning)\b", query.casefold())
    )
    topic_terms = query_terms - {"act", "code", "law", "section", "provision"}
    ranked_sources: list[tuple[float, dict[str, Any]]] = []
    topic_matches: list[tuple[float, dict[str, Any]]] = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        words = re.findall(r"[a-z0-9]+", text.casefold())
        frequencies = {term: words.count(term) for term in query_terms}
        # Blend semantic similarity with lexical matches. The length normalization
        # prevents long chunks from winning only because they repeat a query word.
        lexical_score = sum(min(count, 5) for count in frequencies.values()) / max(len(words) ** 0.5, 1)
        heading = str(metadata.get("heading") or "").casefold()
        lexical_score += 0.5 * sum(term in heading for term in query_terms)
        section_label = citation_section(metadata)
        if section_label and str(section_label).casefold().startswith(("section ", "article ")):
            lexical_score += 0.15
        score = (1.0 / (1.0 + float(distance))) + (0.25 * lexical_score)
        if definition_intent and re.search(
            r"\bis said to (?:commit|be)\b|\bmeans\b|\bshall mean\b",
            text,
            re.IGNORECASE,
        ):
            score += 0.5
        source = {
            "text": text,
            "title": metadata.get("title"),
            "document_type": metadata.get("document_type"),
            "jurisdiction": metadata.get("jurisdiction"),
            "section": section_label,
            "heading": metadata.get("heading"),
            "paragraph": metadata.get("paragraph_number"),
            "pages": {
                "start": metadata.get("page_start"),
                "end": metadata.get("page_end"),
            },
            "source_url": metadata.get("source_url"),
            "distance": distance,
            "score": score,
        }
        ranked_sources.append((score, source))
        searchable_text = f"{text} {heading} {section_label or ''}".casefold()
        if any(re.search(rf"\b{re.escape(term)}\b", searchable_text) for term in topic_terms):
            topic_matches.append((score, source))
    if topic_matches:
        ranked_sources = topic_matches
    ranked_sources.sort(key=lambda item: item[0], reverse=True)
    return [source for _, source in ranked_sources[:limit]]
