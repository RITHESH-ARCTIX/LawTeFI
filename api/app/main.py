
import re
from typing import Any, Literal
from uuid import UUID
from .cross_reference import CONCORDANCE_MAP, parse_cross_reference_query, router as cross_reference_router
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg
from psycopg.types.json import Jsonb

from .auth import get_current_user, router as auth_router
from .database import connect_database
from .documents import router as documents_router, retrieve_user_document
from .generation import generate_document_answer, generate_grounded_answer
from .retrieval import get_statute_section, list_corpus_sources, retrieve

app = FastAPI(title="LawTeFI API", version="0.1.0")
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(cross_reference_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    conversation_id: UUID | None = None
    document_id: UUID | None = None


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=3, max_length=8000)
    limit: int = Field(default=5, ge=1, le=10)


class SavedWorkCreate(BaseModel):
    work_type: Literal["case_brief", "judgment_analysis", "note", "draft", "other"]
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(default="", max_length=100000)
    citations: list[dict[str, Any]] = Field(default_factory=list, max_length=10)


class SavedWorkUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(max_length=100000)


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "lawtefi-api",
        "rag": "local_retrieval_and_generation",
        "pipeline_revision": "shared-legal-source-catalog-v13",
    }


@app.get("/api/v1/conversations")
def list_conversations(current_user: dict[str, object] = Depends(get_current_user)) -> list[dict[str, object]]:
    with connect_database() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.document_id, d.title AS document_title, c.created_at, c.updated_at
            FROM conversations c
            LEFT JOIN user_documents d ON d.id = c.document_id
            WHERE c.owner_user_id = %s
            ORDER BY c.updated_at DESC
            LIMIT 100
            """,
            (current_user["id"],),
        ).fetchall()
    return rows


@app.get("/api/v1/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: UUID,
    current_user: dict[str, object] = Depends(get_current_user),
) -> list[dict[str, object]]:
    with connect_database() as connection:
        owned = connection.execute(
            "SELECT id FROM conversations WHERE id = %s AND owner_user_id = %s",
            (conversation_id, current_user["id"]),
        ).fetchone()
        if owned is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        rows = connection.execute(
            """
            SELECT id, role, content, status, citations, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at, id
            """,
            (conversation_id,),
        ).fetchall()
    return rows


@app.get("/api/v1/saved-work")
def list_saved_work(current_user: dict[str, object] = Depends(get_current_user)) -> list[dict[str, object]]:
    with connect_database() as connection:
        rows = connection.execute(
            """
            SELECT id, work_type, title, content, citations, created_at, updated_at
            FROM saved_work
            WHERE owner_user_id = %s
            ORDER BY updated_at DESC
            LIMIT 200
            """,
            (current_user["id"],),
        ).fetchall()
    return rows


@app.post("/api/v1/saved-work", status_code=status.HTTP_201_CREATED)
def create_saved_work(
    request: SavedWorkCreate,
    current_user: dict[str, object] = Depends(get_current_user),
) -> dict[str, object]:
    try:
        with connect_database() as connection:
            row = connection.execute(
                """
                INSERT INTO saved_work (owner_user_id, work_type, title, content, citations)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, work_type, title, content, citations, created_at, updated_at
                """,
                (
                    current_user["id"], request.work_type, request.title.strip(),
                    request.content, Jsonb(request.citations),
                ),
            ).fetchone()
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="The saved-work database request failed.") from exc
    return row


@app.patch("/api/v1/saved-work/{work_id}")
def update_saved_work(
    work_id: UUID,
    request: SavedWorkUpdate,
    current_user: dict[str, object] = Depends(get_current_user),
) -> dict[str, object]:
    try:
        with connect_database() as connection:
            row = connection.execute(
                """
                UPDATE saved_work
                SET title = %s, content = %s, updated_at = now()
                WHERE id = %s AND owner_user_id = %s
                RETURNING id, work_type, title, content, citations, created_at, updated_at
                """,
                (request.title.strip(), request.content, work_id, current_user["id"]),
            ).fetchone()
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="The saved-work database request failed.") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Saved item not found.")
    return row


@app.delete("/api/v1/saved-work/{work_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved_work(
    work_id: UUID,
    current_user: dict[str, object] = Depends(get_current_user),
) -> None:
    try:
        with connect_database() as connection:
            row = connection.execute(
                "DELETE FROM saved_work WHERE id = %s AND owner_user_id = %s RETURNING id",
                (work_id, current_user["id"]),
            ).fetchone()
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="The saved-work database request failed.") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Saved item not found.")


@app.post("/api/v1/retrieve")
def retrieve_sources(request: RetrievalRequest) -> dict[str, object]:
    """Find relevant legal passages; this endpoint does not generate legal advice."""
    try:
        sources = retrieve(request.question, request.limit)
    except RuntimeError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "question": request.question,
        "sources": sources,
        "message": "This endpoint returns passages only. Use POST /api/v1/chat for a locally generated draft answer.",
    }


@app.get("/api/v1/sources")
def list_sources(current_user: dict[str, object] = Depends(get_current_user)) -> list[dict[str, object]]:
    """List versioned source documents currently indexed in the shared corpus."""
    try:
        return list_corpus_sources()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/chat")
def chat(
    request: ChatRequest,
    current_user: dict[str, object] = Depends(get_current_user),
) -> dict[str, object]:
    """Retrieve local legal evidence, then generate a source-cited draft locally."""
    if request.document_id:
        with connect_database() as connection:
            document = connection.execute(
                "SELECT id FROM user_documents WHERE id = %s AND owner_user_id = %s",
                (request.document_id, current_user["id"]),
            ).fetchone()
        if document is None:
            raise HTTPException(status_code=404, detail="Judgment not found in your workspace.")

    with connect_database() as connection:
        if request.conversation_id:
            conversation = connection.execute(
                "SELECT id, document_id FROM conversations WHERE id = %s AND owner_user_id = %s",
                (request.conversation_id, current_user["id"]),
            ).fetchone()
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found.")
            if conversation["document_id"] != request.document_id:
                raise HTTPException(status_code=409, detail="Start a new conversation to change its source document.")
            conversation_id = request.conversation_id
        else:
            conversation = connection.execute(
                """
                INSERT INTO conversations (owner_user_id, title, document_id)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (current_user["id"], request.question[:80], request.document_id),
            ).fetchone()
            conversation_id = conversation["id"]
        connection.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
            (conversation_id, request.question),
        )

    try:
        if request.document_id:
            sources = retrieve_user_document(
                current_user["id"], request.document_id, request.question, limit=8
            )
            result = generate_document_answer(request.question, sources)
        else:
            broad_topic_question = bool(
                re.search(r"\b(?:what\s+does\b.*\bsay\s+about|tell\s+me\s+about|explain)\b", request.question, re.I)
            )
            bns_section_303_question = bool(
                re.search(r"\bsection\s*303(?:\s*\(\s*[12]\s*\))?\b", request.question, re.I)
                and re.search(r"\b(?:bharatiya\s+nyaya\s+sanhita|bns|theft)\b", request.question, re.I)
            )
            sources = retrieve(request.question, limit=5)
            if (
                broad_topic_question
                and re.search(r"\btheft\b", request.question, re.I)
                and re.search(r"\b(?:bharatiya\s+nyaya\s+sanhita|bns)\b", request.question, re.I)
            ) or bns_section_303_question:
                section_sources = get_statute_section("Bharatiya Nyaya Sanhita, 2023", "303")
                if section_sources:
                    sources = section_sources
            if broad_topic_question and sources:
                main_section = sources[0].get("section")
                if main_section:
                    sources = [source for source in sources if source.get("section") == main_section]
            # Bridge historical IPC / CrPC / IEA queries to operative BNS / BNSS / BSA passages
            if not sources:
                concordance_key = parse_cross_reference_query(request.question)
                if concordance_key and concordance_key in CONCORDANCE_MAP:
                    mapping = CONCORDANCE_MAP[concordance_key]
                    matched_sources = retrieve(mapping["search_query"], limit=3)
                    if matched_sources:
                        sources = matched_sources
            result = generate_grounded_answer(request.question, sources)
    except RuntimeError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result["question"] = request.question
    result["conversation_id"] = str(conversation_id)
    with connect_database() as connection:
        connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, status, citations)
            VALUES (%s, 'assistant', %s, %s, %s)
            """,
            (
                conversation_id,
                result.get("answer") or result.get("message", ""),
                result.get("status"),
                Jsonb(result.get("citations") or []),
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = %s AND owner_user_id = %s",
            (conversation_id, current_user["id"]),
        )
    return result
